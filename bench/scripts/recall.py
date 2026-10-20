"""Answer recall without reasoning (memorisation probe, NOTES §5). gpt-oss-120b gets the official solve prompt, an empty
analysis channel and a final channel opened with "\\boxed{", and samples n short answers (official T/top_p/top_k).
recall = fraction equal to the reference answer. On the bench pool, problems published before the model's cutoff
(AIME/HMMT 2023-2024) should be recalled far more often than 2025-2026 ones; that checks the probe itself.
  python3 -I bench/scripts/recall.py <port> [n]     # writes bench/runs/recall/gpt-oss-120b.jsonl
"""
import asyncio
import json
import os
import re
import sys
import urllib.request

from transformers import AutoTokenizer

ROOT = "/user/wangzhiyao/projects/interp"
MODEL = "/user/wangzhiyao/models/aimo/openai/gpt-oss-120b"
SYSTEM = open(f"{ROOT}/ext/baselines/prompts/solve.txt").read()
FORCE = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>\\boxed{"


def items():
    ans = {r["id"]: r["answer"] for r in map(json.loads, open(f"{ROOT}/bench/train36_answers.jsonl"))}
    for r in map(json.loads, open(f"{ROOT}/bench/train36_input.jsonl")):
        yield {"id": r["id"], "kind": "train36", "source": "train36", "problem": r["problem"], "answer": ans[r["id"]]}
    for r in map(json.loads, open(f"{ROOT}/bench/pool/pool_input.jsonl")):
        yield {"id": r["id"], "kind": "pool", "source": r["source"], "problem": r["problem"], "answer": r["answer"]}


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=600) as r:
        return json.loads(r.read())


def norm(s):
    s = s.strip().replace(",", "").replace("$", "")
    m = re.fullmatch(r"-?\d+(\.0+)?", s)
    return str(int(float(s))) if m else s


async def main(port, n):
    tok = AutoTokenizer.from_pretrained(MODEL)
    out_path = f"{ROOT}/bench/runs/recall/gpt-oss-120b.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sem = asyncio.Semaphore(64)

    async def one(it, effort):
        ids = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["problem"]}],
                                      add_generation_prompt=True, reasoning_effort=effort, return_dict=False)
        ids = list(ids) + tok.encode(FORCE, add_special_tokens=False)
        body = {"model": "gpt-oss-120b", "prompt": ids, "n": n, "max_tokens": 16, "temperature": 1.0, "top_p": 0.95,
                "top_k": 40, "stop": ["}"]}
        async with sem:
            r = await asyncio.to_thread(post, port, body)
        got = [norm(c["text"].split("}")[0]) for c in r["choices"]]
        ref = norm(it["answer"])
        return {"id": it["id"], "kind": it["kind"], "source": it["source"], "effort": effort, "answers": got,
                "recall": sum(g == ref for g in got) / len(got)}

    res = await asyncio.gather(*(one(it, e) for it in items() for e in ("low", "medium")))
    with open(out_path, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(res)} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 32))
