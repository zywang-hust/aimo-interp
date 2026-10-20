"""Check submission/fragility/rollout.py against the sglang samples of bench/runs/train36all (same prompt, effort).

  CUDA_VISIBLE_DEVICES=<id> EVAL_EXTRA_SITE=engine/evalrt/kernels-overlay engine/.venv-eval/bin/python -I \
      engine/evalrt/rollout_check.py <model-path> <effort> <batch> <max_new> id [id ...]
Prints, per problem, the fraction of samples longer than 2048/3072 tokens here and in the sglang reference, and the
pooled fractions at the end. FEAT_LAYER=<L> also passes feat_layer (the probe path; sampling must not change).
"""
import json
import os
import sys

if os.environ.get("EVAL_EXTRA_SITE"):
    sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])
ROOT = "/user/wangzhiyao/projects/interp"
sys.path.insert(0, f"{ROOT}/submission/fragility")

import numpy as np  # noqa: E402
rollouts = __import__(os.environ.get("ROLLOUT_MOD", "rollout")).rollouts  # v1, or rollout_fast (v2)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

path, effort, B, T, ids = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
SOLVE = open(f"{ROOT}/ext/baselines/prompts/solve.txt").read().strip()
probs = {r["id"]: r["problem"] for r in map(json.loads, open(f"{ROOT}/bench/train36_input.jsonl"))}
ref = {}
for r in map(json.loads, open(f"{ROOT}/bench/runs/train36all/out/gpt-oss-120b__{effort}.jsonl")):
    ref.setdefault(r["id"], []).append(r["completion_tokens"])
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto").eval()
pool = []
for pid in ids:
    x = tok.apply_chat_template([{"role": "system", "content": SOLVE}, {"role": "user", "content": probs[pid]}],
                                tokenize=True, add_generation_prompt=True, reasoning_effort=effort)
    x = list(x["input_ids"] if hasattr(x, "keys") else x)
    r = rollouts(tok, model, x, batch=B, max_new=T, **({"feat_layer": int(os.environ["FEAT_LAYER"])} if os.environ.get("FEAT_LAYER") else {}))
    ln = np.array([v if v > 0 else 10**9 for v in r["len"]])
    rf = np.array(ref[pid])
    pool.append((ln, rf))
    print(f"{pid}: P={len(x)} steps {r['steps']} {r['s']}s ({r['setup_s']}s setup, {r['ms_per_step']} ms/step) | "
          f">2048 {np.mean(ln > 2048):.2f} vs {np.mean(rf > 2048):.2f} | >3072 {np.mean(ln > 3072):.2f} vs "
          f"{np.mean(rf > 3072):.2f} | answers {r['answers']}"
          + (f" | check {r['check']}" if r.get("check") else "") + (f" | FALLBACK {r['fallback']}" if r.get("fallback") else ""),
          flush=True)
here, there = np.concatenate([a for a, _ in pool]), np.concatenate([b for _, b in pool])
print(f"pooled over {len(pool)} problems ({len(here)} vs {len(there)} samples): >2048 {np.mean(here > 2048):.3f} vs "
      f"{np.mean(there > 2048):.3f} | >3072 {np.mean(here > 3072):.3f} vs {np.mean(there > 3072):.3f} | "
      f"mean per-problem >2048 {np.mean([np.mean(a > 2048) for a, _ in pool]):.3f} vs "
      f"{np.mean([np.mean(b > 2048) for _, b in pool]):.3f}", flush=True)
