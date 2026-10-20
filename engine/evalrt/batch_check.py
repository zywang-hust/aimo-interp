"""Check per-row prompts in submission/fragility/rollout.py (left padding, per-row masks and positions).
1. one prompt passed once vs as a per-row list: identical tokens (same seed);
2. two problems of different prompt length mixed in one batch vs each alone, near-greedy (temperature 1e-4):
   first position where a row departs from the same row run alone, next to the noise floor of the same prompt at
   batch 4 vs batch 8 (HF itself departs across batch sizes at near-ties, NOTES.md §4.2).
  GPU=<id> engine/evalrt/run_eval.sh engine/evalrt/batch_check.py <model-path> <max_new> id_a id_b
"""
import json
import os
import sys

import torch

ROOT = "/user/wangzhiyao/projects/interp"
if os.environ.get("EVAL_EXTRA_SITE"):
    sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])
sys.path.insert(0, f"{ROOT}/submission/fragility")
from fragility import SOLVE_PROMPT  # noqa: E402
from rollout import rollouts  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def first_diff(a, b):
    n = min(len(a), len(b))
    d = (a[:n] != b[:n]).nonzero()
    return int(d[0]) if len(d) else n


def main():
    path, max_new, ids = sys.argv[1], int(sys.argv[2]), sys.argv[3:5]
    probs = {r["id"]: r["problem"] for r in map(json.loads, open(f"{ROOT}/bench/train36_input.jsonl"))}
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto").eval()

    def ids_of(p):
        x = tok.apply_chat_template([{"role": "system", "content": SOLVE_PROMPT}, {"role": "user", "content": probs[p]}],
                                    tokenize=True, add_generation_prompt=True, reasoning_effort="medium")
        return list(x["input_ids"] if hasattr(x, "keys") else x)
    a, b = ids_of(ids[0]), ids_of(ids[1])
    print(f"prompt lengths {len(a)} {len(b)}", flush=True)

    r1 = rollouts(tok, model, a, batch=4, max_new=max_new, return_toks=True)
    r2 = rollouts(tok, model, [a] * 4, batch=4, max_new=max_new, return_toks=True)
    print(f"1. same prompt, single vs per-row: tokens identical {torch.equal(r1['toks'], r2['toks'])}, "
          f"len {r1['len']} vs {r2['len']}", flush=True)

    g = dict(max_new=max_new, return_toks=True, temperature=1e-4)
    ra, rb = rollouts(tok, model, a, batch=4, **g), rollouts(tok, model, b, batch=4, **g)
    ra8 = rollouts(tok, model, a, batch=8, **g)
    mix = rollouts(tok, model, [a] * 4 + [b] * 4, batch=8, **g)
    print(f"2. near-greedy, max_new {max_new}, first departure from the row run alone (batch 4):")
    print(f"   noise floor, same prompt batch 8: {[first_diff(ra8['toks'][i], ra['toks'][i]) for i in range(4)]}")
    print(f"   mixed, rows of {ids[0]} (P={len(a)}): {[first_diff(mix['toks'][i], ra['toks'][i]) for i in range(4)]}")
    print(f"   mixed, rows of {ids[1]} (P={len(b)}): {[first_diff(mix['toks'][4 + i], rb['toks'][i]) for i in range(4)]}")
    print(f"   lengths alone {ra['len']} {rb['len']} | mixed {mix['len']} | ms/step {mix['ms_per_step']}")
    print(f"   text check, mixed row 4: {tok.decode(mix['toks'][4, :60].tolist())!r}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
