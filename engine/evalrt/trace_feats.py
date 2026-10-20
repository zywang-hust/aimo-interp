"""Teacher-forced mid-reasoning features of gpt-oss traces (what an eval-time rollout would see at step t).

For each kept trace (sglang sample with reasoning text) rebuild prompt + analysis-channel tokens, run one HF forward
over the first max(CKPT) generated tokens, and store at each checkpoint t in CKPT:
  - hidden state of the token at position t at LAYERS (mean over the last 16 tokens up to t) -> H,
  - hidden state averaged over all generated tokens [0, t) at LAYERS -> M,
  - mean / std token entropy and mean chosen-token logprob over generated tokens [0, t) -> E.
CKPT / LAYERS can be overridden with TRACE_CKPT / TRACE_LAYERS (comma lists).
  CUDA_VISIBLE_DEVICES=<id> EVAL_EXTRA_SITE=engine/evalrt/kernels-overlay engine/.venv-eval/bin/python -I \
      engine/evalrt/trace_feats.py <model-path> <samples.jsonl> <input.jsonl> <out.npz> [per_problem] [shard/nshards]
"""
import json
import os
import sys
import time

if os.environ.get("EVAL_EXTRA_SITE"):
    sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"
SOLVE = open(f"{ROOT}/ext/baselines/prompts/solve.txt").read().strip()
CKPT = [int(x) for x in os.environ.get("TRACE_CKPT", "256,512,1024,2048,3072").split(",")]
LAYERS = [int(x) for x in os.environ.get("TRACE_LAYERS", "12,18,24,30,36").split(",")]

path, samples, inputs, out = sys.argv[1:5]
per = int(sys.argv[5]) if len(sys.argv) > 5 else 2
shard, nsh = map(int, (sys.argv[6] if len(sys.argv) > 6 else "0/1").split("/"))
probs = {r["id"]: r["problem"] for r in map(json.loads, open(inputs))}
rows, cnt = [], {}
for r in map(json.loads, open(samples)):
    if r.get("reasoning") and r["id"] in probs and cnt.get(r["id"], 0) < per:
        cnt[r["id"]] = cnt.get(r["id"], 0) + 1
        rows.append(r)
rows = rows[shard::nsh]
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto").eval()
H, M, E, meta = [], [], [], []
t0 = time.time()
for i, r in enumerate(rows):
    msgs = [{"role": "system", "content": SOLVE}, {"role": "user", "content": probs[r["id"]]}]
    p = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, reasoning_effort="medium")
    p = list(p["input_ids"] if hasattr(p, "keys") else p)
    g = tok("<|channel|>analysis<|message|>" + r["reasoning"], add_special_tokens=False)["input_ids"][:max(CKPT)]
    ids = torch.tensor([p + g], device=model.device)
    with torch.inference_mode():
        o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        lp = torch.log_softmax(o.logits[0, len(p) - 1:-1].float(), -1)
        ent = -(lp.exp() * lp).sum(-1)
        chosen = lp.gather(-1, ids[0, len(p):, None])[:, 0]
        h, mm, e = [], [], []
        for t in CKPT:
            tt = min(t, len(g))
            pos = len(p) + tt - 1
            h.append(torch.stack([o.hidden_states[L][0, max(len(p), pos - 15):pos + 1].float().mean(0)
                                  for L in LAYERS]).cpu().numpy())
            mm.append(torch.stack([o.hidden_states[L][0, len(p):pos + 1].float().mean(0)
                                   for L in LAYERS]).cpu().numpy())
            e.append([ent[:tt].mean().item(), ent[:tt].std().item(), chosen[:tt].mean().item(), tt])
    H.append(np.stack(h).astype(np.float16))
    M.append(np.stack(mm).astype(np.float16))
    E.append(e)
    meta.append({"id": r["id"], "k": r.get("k"), "len": r["completion_tokens"], "answer": str(r.get("answer")),
                 "finish": r.get("finish"), "n_gen": len(g)})
    del o
    if i % 50 == 0:
        print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
np.savez(out, H=np.stack(H), M=np.stack(M), E=np.array(E), meta=json.dumps(meta), ckpt=CKPT, layers=LAYERS)
print("done", len(rows), f"{time.time() - t0:.0f}s", flush=True)
