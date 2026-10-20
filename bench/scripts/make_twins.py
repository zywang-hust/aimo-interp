"""Create number-modified 'twins' of competition problems, mimicking the official construction
(train-main-v2 pairs AIME/AIMO originals with versions whose numbers were changed).

The generator proposes a twin plus its own answer. Twins are NOT trusted until verified downstream:
sample strong solvers (sample.py) on the twin and keep it only if the solver consensus matches the proposed
answer (see verify step in the bench pipeline).

Output jsonl: {id, parent, j, problem, proposed_answer, change}
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import sys
import urllib.request

PROMPT = """You are helping build a math benchmark. Below is a competition problem with its known answer.
Create a NEW variant by changing one to three of the numerical parameters (or a modulus / exponent / bound),
keeping the same structure, wording and difficulty. The variant must still have a single well-defined
non-negative integer answer, and that answer must be DIFFERENT from the original answer. Avoid changes that
make the problem degenerate or trivially easy. Then solve the variant carefully.

Original problem:
{problem}

Original answer: {answer}

Reply with a JSON object only, no markdown fences:
{{"problem": "<full text of the variant, same formatting style as the original>",
  "change": "<one line: what was changed>",
  "answer": <integer answer of the variant>}}"""


def chat(base_url, model, prompt, max_tokens, temperature, extra):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature}
    payload.update(extra or {})
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=7200) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "problem" not in d or not re.fullmatch(r"-?\d+", str(d.get("answer", "")).strip()):
        return None
    return d


async def one(sem, args, row, j, out_f, lock):
    loop = asyncio.get_running_loop()
    rec = {"parent": row["id"], "j": j}
    async with sem:
        try:
            txt = await loop.run_in_executor(None, chat, args.url, args.model,
                                             PROMPT.format(problem=row["problem"], answer=row["answer"]),
                                             args.max_tokens, 0.8, json.loads(args.extra or "{}"))
            d = parse(txt)
            if d and str(d["answer"]).strip() != str(row["answer"]):
                rec.update(problem=d["problem"].strip(), proposed_answer=str(d["answer"]).strip(),
                           change=d.get("change", ""))
                rec["id"] = hashlib.sha1(re.sub(r"\s+", " ", rec["problem"]).encode()).hexdigest()[:8]
            else:
                rec["error"] = "unparseable or same answer"
        except Exception as e:
            rec["error"] = str(e)[:300]
    async with lock:
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()


async def main_async(args):
    # the default executor caps at min(32, cpus+4) threads, which silently capped concurrency at 32
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=args.concurrency + 8))
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            if "error" not in r:
                done.add((r["parent"], r["j"]))
    todo = [(r, j) for r in rows for j in range(args.k) if (r["id"], j) not in done]
    print(f"{len(todo)} twins to generate", file=sys.stderr)
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    with open(args.out, "a") as out_f:
        await asyncio.gather(*(one(sem, args, r, j, out_f, lock) for r, j in todo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl {id, problem, answer}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=64)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
