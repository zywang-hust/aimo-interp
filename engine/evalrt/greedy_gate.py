"""Greedy gate: rollout_fast (greedy=True) vs stock HF generate (do_sample=False, eager attention, DynamicCache).

  run_eval.sh engine/evalrt/greedy_gate.py <model> <effort> <batch> <n_tokens> id [id ...]
Exact token equality over hundreds of steps is not attainable by any kernel change: HF eager itself departs from its
own batch-1 output when run at batch B (different GEMM / MoE kernels per M). So per problem this prints the first
departure position of
  HF batch B vs HF batch 1   (noise floor of the reference itself)
  fast vs HF batch 1, fast vs HF batch B
  fast with the default MoE flags vs HF batch B   (GATE_MOE_DEFAULT=1)
and at each departure the reference's own top-1 vs top-2 logit margin there (a departure at a near-tie is rounding
noise; at a clear margin it is a bug).
"""
import json
import os
import sys

sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])
ROOT = "/user/wangzhiyao/projects/interp"
sys.path.insert(0, f"{ROOT}/submission/fragility")
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

mod = __import__(os.environ.get("ROLLOUT_MOD", "rollout_fast"))
path, effort, B, T, ids = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
SOLVE = open(f"{ROOT}/ext/baselines/prompts/solve.txt").read().strip()
probs = {r["id"]: r["problem"] for r in map(json.loads, open(f"{ROOT}/bench/train36_input.jsonl"))}
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto").eval()
print("attn_implementation:", model.config._attn_implementation, flush=True)


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def hf(x, nb):
    with torch.inference_mode():
        o = model.generate(torch.tensor([x] * nb, device=model.device), do_sample=False, max_new_tokens=T,
                           attention_mask=torch.ones(nb, len(x), dtype=torch.long, device=model.device),
                           output_logits=True, return_dict_in_generate=True)
    toks = o.sequences[:, len(x):].tolist()
    top2 = torch.stack([lg.float().topk(2, dim=-1).values for lg in o.logits], 1)  # (nb, steps, 2)
    return toks, (top2[..., 0] - top2[..., 1]).tolist()


def report(name, ref, margin, rows):
    d = [first_diff(ref, r) for r in rows]
    i = min(d)
    m = f" (reference top1-top2 margin there {margin[i]:.3f})" if i < min(len(ref), T) else ""
    print(f"   {name:28s} first departure per row {d}{m}", flush=True)
    return i


for pid in ids:
    x = tok.apply_chat_template([{"role": "system", "content": SOLVE}, {"role": "user", "content": probs[pid]}],
                                tokenize=True, add_generation_prompt=True, reasoning_effort=effort)
    x = list(x["input_ids"] if hasattr(x, "keys") else x)
    (h1,), (m1,) = (lambda t: ([t[0][0]], [t[1][0]]))(hf(x, 1))
    hb, mb = hf(x, B)
    print(f"{pid} P={len(x)} batch {B}, {T} tokens (HF batch-1 length {len(h1)})", flush=True)
    report(f"HF batch {B} vs HF batch 1", h1, m1, hb)
    runs = [("fast", None)] + ([("fast default-MoE", "1")] if os.environ.get("GATE_MOE_DEFAULT") else [])
    for name, env in runs:
        if env:
            os.environ["ROLLOUT_MOE_DEFAULT"] = env
        r = mod.rollouts(tok, model, x, batch=B, max_new=T, greedy=True)
        os.environ.pop("ROLLOUT_MOE_DEFAULT", None)
        report(f"{name} vs HF batch 1", h1, m1, r["toks"])
        report(f"{name} vs HF batch {B}", hb[0], mb[0], r["toks"])
