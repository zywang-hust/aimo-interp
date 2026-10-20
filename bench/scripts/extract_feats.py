"""Prefill hidden-state features for white-box probes, extracted in the eval-runtime replica.

  CUDA_VISIBLE_DEVICES=<id> engine/.venv-eval/bin/python -I bench/scripts/extract_feats.py \
      --model-key qwen3.5-4b --input bench/train36_input.jsonl --out bench/feats/train36/qwen3.5-4b__default.npz
  (gpt-oss: --effort high and EVAL_EXTRA_SITE=engine/evalrt/kernels-overlay, see engine/README.md)

The prompt is exactly what the model sees when solving: solve.txt as system message, problem as user message,
chat template with the generation prompt (gpt-oss: reasoning_effort rendered into the template).
Saved per problem, for every layer (embeddings + each block):
  last  hidden state at the final prompt token (where generation starts)       [N, L+1, d] float16
  mean  mean hidden state over the user-problem span (approximated as all tokens after the system prompt)
Plus the next-token entropy at the last position. One problem per forward (no padding effects).
"""
import argparse
import json
import os
import sys
import time

if os.environ.get("EVAL_EXTRA_SITE"):
    sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_KEYS  # noqa: E402
from sample import SOLVE_PROMPT  # noqa: E402

PATHS = {k: f"/user/wangzhiyao/models/aimo/{v}" for k, v in MODEL_KEYS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted(PATHS))
    ap.add_argument("--input", required=True, help="jsonl {id, problem}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", default=None, help="gpt-oss reasoning effort")
    args = ap.parse_args()

    path = PATHS[args.model_key]
    tok_cls = PreTrainedTokenizerFast if args.model_key == "dsr1-qwen3-8b" else AutoTokenizer  # HF 5.13 bug
    tok = tok_cls.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
    model.eval()
    kw = {"reasoning_effort": args.effort} if args.effort else {}

    def encode(problem):
        msgs = [{"role": "system", "content": SOLVE_PROMPT}, {"role": "user", "content": problem}]
        x = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
        return list(x["input_ids"] if hasattr(x, "keys") else x)

    n_sys = len(encode(""))  # tokens of the template with an empty problem ~ everything but the problem
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    ids, last, mean, ent, ntok = [], [], [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for r in rows:
            x = encode(r["problem"])
            out = model(input_ids=torch.tensor([x], device=model.device), output_hidden_states=True)
            hs = torch.stack(out.hidden_states)[:, 0]  # [L+1, T, d]
            span = slice(max(n_sys - 8, 0), len(x))  # problem tokens (+ a few template tokens); crude but fixed
            last.append(hs[:, -1].float().cpu().numpy().astype(np.float16))
            mean.append(hs[:, span].float().mean(1).cpu().numpy().astype(np.float16))
            p = torch.softmax(out.logits[0, -1].float(), -1)
            ent.append(float(-(p * torch.log(p + 1e-12)).sum()))
            ids.append(r["id"])
            ntok.append(len(x))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, ids=np.array(ids), last=np.stack(last), mean=np.stack(mean), entropy=np.array(ent),
             ntok=np.array(ntok))
    print(json.dumps({"model": args.model_key, "effort": args.effort, "n": len(ids), "layers": len(last[0]),
                      "d": int(last[0].shape[-1]), "secs": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
