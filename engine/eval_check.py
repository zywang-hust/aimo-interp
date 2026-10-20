"""Eval-runtime replica check (run with engine/.venv-eval/bin/python -I, CUDA_VISIBLE_DEVICES=<one id>).

  python -I engine/eval_check.py <model-path> [--batch 32] [--seq 300] [--fix-tokenizer]

Loads the model the way the official harness does (AutoModelForCausalLM.from_pretrained(path,
device_map="auto")), then times one forward pass over a batch of `batch` prompts of `seq` tokens.
Prints one JSON line.
"""
import argparse, json, os, sys, time

# Optional extra site dir (used only to test gpt-oss with the `kernels` package, which the official
# Dockerfile does not install). Unset = exact replica of the official runtime.
if os.environ.get("EVAL_EXTRA_SITE"):
    sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seq", type=int, default=300)
ap.add_argument("--fix-tokenizer", action="store_true", help="load with PreTrainedTokenizerFast (DSR1 workaround)")
a = ap.parse_args()

out = {"model": a.path.split("/aimo/")[-1], "torch": torch.__version__, "transformers": transformers.__version__,
       "gpu": torch.cuda.get_device_name(0), "n_gpu": torch.cuda.device_count()}
tok = (PreTrainedTokenizerFast if a.fix_tokenizer else AutoTokenizer).from_pretrained(a.path)
out["tokenizer"] = type(tok).__name__

torch.cuda.reset_peak_memory_stats()
t = time.time()
model = AutoModelForCausalLM.from_pretrained(a.path, device_map="auto")
torch.cuda.synchronize()
out["load_s"] = round(time.time() - t, 1)
out["class"] = type(model).__name__
out["dtype"] = str(model.dtype)
out["device_map_devices"] = sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()})
out["gpu_mem_after_load_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
qc = getattr(model.config, "quantization_config", None)
if qc is not None:
    qd = qc if isinstance(qc, dict) else qc.to_dict()
    out["quant"] = {k: qd.get(k) for k in ("quant_method", "dequantize")}
    names = {type(m).__name__ for m in model.modules()}
    out["expert_modules"] = sorted(n for n in names if "xpert" in n or "Mxfp4" in n or "MoE" in n)

# 32 distinct prompts of exactly `seq` tokens, built from varied math text through the chat template.
base = ("Let $a_1, a_2, \\ldots$ be a sequence with $a_1 = {k}$ and $a_{{n+1}} = a_n^2 - {k} a_n + 7$ for all $n \\ge 1$. "
        "Find the remainder when $a_{{2025}}$ is divided by $1000$. Also determine all primes $p < 100$ dividing "
        "$a_{k}$, and the number of lattice points inside the circle $x^2 + y^2 = {k}00$. ")
ids = []
for k in range(a.batch):
    msgs = [{"role": "user", "content": (base.format(k=k + 3)) * 6}]
    x = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    x = x["input_ids"] if hasattr(x, "keys") else x
    assert len(x) >= a.seq, len(x)
    ids.append(x[-a.seq:])  # keep the generation-prompt tail
inp = torch.tensor(ids, device=model.device)
attn = torch.ones_like(inp)
with torch.inference_mode():
    torch.cuda.synchronize(); t = time.time()
    model(input_ids=inp, attention_mask=attn)          # first call (includes lazy kernel init)
    torch.cuda.synchronize(); out["fwd_first_s"] = round(time.time() - t, 3)
    times = []
    for _ in range(3):
        t = time.time(); r = model(input_ids=inp, attention_mask=attn); torch.cuda.synchronize()
        times.append(time.time() - t)
    out["fwd_warm_s"] = round(min(times), 3)
    out["logits_finite"] = bool(torch.isfinite(r.logits[:, -1].float()).all())
    out["top1_last"] = tok.decode([int(r.logits[0, -1].argmax())])
out["batch_x_seq"] = f"{a.batch}x{a.seq}"
out["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
print("RESULT " + json.dumps(out), flush=True)
