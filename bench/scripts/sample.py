"""Resumable async sampler against OpenAI-compatible servers (sglang / vLLM).

Input jsonl rows: {"id": str, "problem": str, ...}. Output jsonl rows: one per (id, sample_idx), holding the
extracted answer, finish_reason and token counts; raw text optional (--keep-text). Re-running skips done pairs.

Defaults follow the official generation settings: temperature 1.0, top_k 40, top_p 0.95, and the official
solve prompt from aimo-interp/baselines (prompts/solve.txt) as the system message.

Example:
  python -I sample.py --input q.jsonl --out s.jsonl --model qwen35-4b \
      --base-url http://127.0.0.1:31000/v1 --n 64 --max-tokens 32768 --concurrency 256
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
import sys
import time
import urllib.request

SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, then output your final "
    "answer on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer "
    "value or NaN if the given problem does not have a solution."
)

_ANSWER_RE = re.compile(r"ANSWER\s*[:：]\s*\**\s*\$?\s*([^\n$*]+)", re.IGNORECASE)


def _last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def normalize(ans):
    """Map a raw answer string to a canonical form: an int string, 'NaN', or None (unparseable)."""
    if ans is None:
        return None
    s = ans.strip().strip(".").strip()
    s = re.sub(r"\\(text|mathrm|textbf)\{([^}]*)\}", r"\2", s)
    s = s.replace("{,}", "").replace("\\,", "").replace(",", "").replace("\\!", "").replace(" ", "").strip("$")
    if re.fullmatch(r"(?i)nan|none|nosolution|\\text\{nan\}", s):
        return "NaN"
    m = re.fullmatch(r"[+-]?\d+(\.0+)?", s)
    if m:
        return str(int(float(s)))
    return None


def extract_answer(text):
    """Official format first (last 'ANSWER:'), then a \\boxed{} or lone number right after it, then the last
    \\boxed{} anywhere. Returns (raw, normalized)."""
    if not text:
        return None, None
    hits = list(re.finditer(r"ANSWER\s*[:：]", text, re.IGNORECASE))
    if hits:
        tail = text[hits[-1].end():hits[-1].end() + 300]
        m = _ANSWER_RE.search(text[hits[-1].start():])
        cands = [m.group(1) if m else None, _last_boxed(tail)]
        num = re.search(r"^[\s\\\[\]()$*]*(-?\d[\d,{}]*|NaN)(?![\d/.^])", tail, re.IGNORECASE)
        cands.append(num.group(1) if num else None)
        for c in cands:
            if normalize(c) is not None:
                return c, normalize(c)
    raw = _last_boxed(text)
    if raw is None and hits:
        raw = tail.strip()[:80]
    return raw, normalize(raw)


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.load(r)


async def worker(name, queue, args, base_urls, out_f, lock, stats):
    loop = asyncio.get_running_loop()
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        row, k = item
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": row["problem"]})
        payload = {
            "model": args.model, "messages": messages, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "seed": hash((row["id"], k, args.seed)) % (2**31),
        }
        if args.reasoning_effort:
            payload["reasoning_effort"] = args.reasoning_effort
            payload["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
        if args.extra:
            payload.update(json.loads(args.extra))
        url = random.choice(base_urls).rstrip("/") + "/chat/completions"
        t0 = time.time()
        rec = {"id": row["id"], "k": k, "model": args.model, "effort": args.reasoning_effort or "default"}
        for attempt in range(4):
            try:
                resp = await loop.run_in_executor(None, post, url, payload, args.timeout)
                ch = resp["choices"][0]
                msg = ch["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                raw, ans = extract_answer(content)
                if ans is None and not content:  # some parsers leave everything in reasoning
                    raw, ans = extract_answer(reasoning)
                rec.update(raw_answer=raw, answer=ans, finish=ch.get("finish_reason"),
                           completion_tokens=resp.get("usage", {}).get("completion_tokens"),
                           prompt_tokens=resp.get("usage", {}).get("prompt_tokens"), secs=round(time.time() - t0, 1))
                if args.keep_text:
                    rec.update(content=content, reasoning=reasoning)
                break
            except Exception as e:  # retry transient server errors
                rec.update(error=str(e)[:300])
                await asyncio.sleep(5 * (attempt + 1))
        async with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["done"] += 1
            stats["tok"] += rec.get("completion_tokens") or 0
        queue.task_done()


async def main_async(args):
    # the default executor caps at min(32, cpus+4) threads, which silently capped concurrency at 32
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=args.concurrency + 8))
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l)
                if "answer" in r:  # rows with only an error are retried
                    done.add((r["id"], r["k"]))
            except json.JSONDecodeError:
                pass
    todo = [(r, k) for k in range(args.n) for r in rows if (r["id"], k) not in done]
    for extra in args.skip_done_in or []:  # e.g. the forward run's output when this is a --reverse helper
        if os.path.exists(extra):
            other = {(r["id"], r["k"]) for r in map(json.loads, open(extra)) if "answer" in r}
            todo = [(r, k) for r, k in todo if (r["id"], k) not in other]
    if args.reverse:  # second replica working from the other end into its own --out file
        todo = todo[::-1]
    print(f"{len(rows)} problems x {args.n} samples: {len(done)} done, {len(todo)} to go", file=sys.stderr)
    queue = asyncio.Queue()
    for it in todo:
        queue.put_nowait(it)
    for _ in range(args.concurrency):
        queue.put_nowait(None)
    lock, stats, t0 = asyncio.Lock(), {"done": 0, "tok": 0}, time.time()
    with open(args.out, "a") as out_f:
        ws = [asyncio.create_task(worker(i, queue, args, args.base_url, out_f, lock, stats))
              for i in range(args.concurrency)]

        async def report():
            while True:
                await asyncio.sleep(60)
                dt = time.time() - t0
                print(f"[{dt/60:.1f} min] {stats['done']}/{len(todo)} done, {stats['tok']/dt:.0f} tok/s", file=sys.stderr)
        rep = asyncio.create_task(report())
        await asyncio.gather(*ws)
        rep.cancel()
    print(f"finished {stats['done']} in {(time.time()-t0)/60:.1f} min", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--base-url", required=True, nargs="+", help="one or more .../v1 endpoints (load-balanced)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--reasoning-effort", default=None, help="gpt-oss: low/medium/high")
    ap.add_argument("--system-prompt", default=SOLVE_PROMPT, help="'' to disable")
    ap.add_argument("--extra", default=None, help="JSON merged into the request payload")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="process (k, row) pairs from the end")
    ap.add_argument("--skip-done-in", nargs="*", help="other output files whose done pairs are skipped at start")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
