"""Check the in-graph probe feature of submission/fragility/rollout.py (v1; ROLLOUT_MOD=rollout_fast for v2)
against teacher forcing.

1. HF hidden_states[L] == output of decoder layer L-1 (what v1's hook grabs, and what v2 adds up in its decode step);
2. rollouts(..., feat_layer=L): for samples still running at the cap, the captured mean of the last 16 generated
   tokens equals hidden_states[L] averaged over the same tokens in one teacher-forced forward (bf16 noise only).
  [ROLLOUT_MOD=rollout_fast] GPU=<id> engine/evalrt/run_eval.sh engine/evalrt/feat_check.py <model-path> <pid> [max_new] [batch]
"""
import json
import os
import sys

if os.environ.get("EVAL_EXTRA_SITE"):
    sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])
ROOT = "/user/wangzhiyao/projects/interp"
sys.path.insert(0, f"{ROOT}/submission/fragility")

import torch  # noqa: E402
rollouts = __import__(os.environ.get("ROLLOUT_MOD", "rollout")).rollouts  # v1, or rollout_fast (v2)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

path, pid = sys.argv[1], sys.argv[2]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
B = int(sys.argv[4]) if len(sys.argv) > 4 else 4
LAYER = 18
SOLVE = open(f"{ROOT}/ext/baselines/prompts/solve.txt").read().strip()
probs = {r["id"]: r["problem"] for r in map(json.loads, open(f"{ROOT}/bench/train36_input.jsonl"))}
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto").eval()
x = tok.apply_chat_template([{"role": "system", "content": SOLVE}, {"role": "user", "content": probs[pid]}],
                            tokenize=True, add_generation_prompt=True, reasoning_effort="medium")
x = list(x["input_ids"] if hasattr(x, "keys") else x)

got = {}
h = model.model.layers[LAYER - 1].register_forward_hook(lambda m, a, o: got.__setitem__("h", o[0] if isinstance(o, tuple) else o))
with torch.inference_mode():
    o = model(input_ids=torch.tensor([x], device=model.device), output_hidden_states=True)
h.remove()
print(f"hook vs hidden_states[{LAYER}]: max abs diff {(got['h'] - o.hidden_states[LAYER]).abs().max().item():.3g}, "
      f"vs [{LAYER + 1}] {(got['h'] - o.hidden_states[LAYER + 1]).abs().max().item():.3g}")

r = rollouts(tok, model, x, batch=B, max_new=T, feat_layer=LAYER, return_toks=True)
print(os.environ.get("ROLLOUT_MOD", "rollout"), "len", r["len"], "steps", r["steps"], "ms/step", r["ms_per_step"],
      "check", r.get("check"), "fallback", r.get("fallback"), flush=True)
import numpy as np  # noqa: E402
pr = np.load(f"{ROOT}/submission/fragility/probe.npz")  # the package's length probe, as fragility.py applies it
probe = lambda v: float((((v.numpy() - pr["mu"]) / pr["sd"] - pr["pmean"]) @ pr["comps"].T) @ pr["coef"] + pr["b"])  # noqa: E731
cos_all, dpr = [], []
mean16 = lambda h: h[0, len(x) + T - 16:len(x) + T].float().mean(0).cpu()  # noqa: E731
cos_of = lambda a, c: torch.nn.functional.cosine_similarity(a, c, dim=0).item()  # noqa: E731
for b in range(B):
    if r["len"][b] > 0:
        continue
    ids = torch.tensor([x + r["toks"][b].tolist()], device=model.device)
    with torch.inference_mode():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    ref, f = mean16(hs[LAYER]), r["feat"][b]
    cos = cos_of(ref, f)
    cos_all.append(cos)
    dpr.append(probe(f) - probe(ref))
    # specificity: the same feature against the neighbouring layers' teacher-forced states
    print(f"sample {b}: cos {cos:.5f}, rel err {((ref - f).norm() / ref.norm()).item():.4f}, norm {ref.norm().item():.1f}"
          f" | vs layer {LAYER - 1} {cos_of(mean16(hs[LAYER - 1]), f):.4f}, {LAYER + 1} {cos_of(mean16(hs[LAYER + 1]), f):.4f}"
          f" | probe {probe(f):.3f} vs teacher-forced {probe(ref):.3f}", flush=True)
    del hs
if cos_all:
    print(f"{len(cos_all)} unfinished samples: cos min {min(cos_all):.5f} mean {sum(cos_all) / len(cos_all):.5f} | "
          f"probe(feat) - probe(teacher-forced): max |d| {max(map(abs, dpr)):.3f}, mean {np.mean(dpr):+.3f}")
