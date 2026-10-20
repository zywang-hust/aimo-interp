"""Minimal async client for engine servers: smoke test and throughput benchmark.

  python engine/client.py smoke --port 30001 [--effort high]
  python engine/client.py bench --port 30001 -n 256 --max-tokens 16384 [--concurrency 64] [--effort low] [--out f.jsonl]

Always sends the official sampling params (T=1.0, top_p=0.95, top_k=40).
"""
import argparse, asyncio, json, statistics, time

import aiohttp

PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
# AIME 2024 I #2 (answer 25) and a few more AIME-style problems for the benchmark mix.
PROBLEMS = [
    ("There exist real numbers $x$ and $y$, both greater than 1, such that "
     "$\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.", "25"),
    ("Find the number of ordered pairs $(m,n)$ of positive integers such that $m^2 n = 20^{20}$.", "231"),
    ("Let $N$ be the greatest four-digit positive integer with the property that whenever one of its "
     "digits is changed to $1$, the resulting number is divisible by $7$. Let $Q$ and $R$ be the "
     "quotient and remainder, respectively, when $N$ is divided by $1000$. Find $Q+R$.", "699"),
    ("Find the remainder when $9 \\times 99 \\times 999 \\times \\cdots \\times "
     "\\underbrace{99\\cdots9}_{\\text{999 9's}}$ is divided by $1000$.", "109"),
]


def last_boxed(text):
    if not text:
        return None
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k].strip()
    return None


async def one(session, url, model, problem, max_tokens, effort, seed=None):
    body = {"model": model, "messages": [{"role": "user", "content": problem + PROMPT_SUFFIX}],
            "temperature": 1.0, "top_p": 0.95, "top_k": 40, "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort
    if seed is not None:
        body["seed"] = seed
    t = time.time()
    async with session.post(url, json=body) as r:
        d = await r.json()
    if "choices" not in d:
        return {"error": d, "latency": time.time() - t}
    c = d["choices"][0]
    msg = c["message"]
    return {"finish": c["finish_reason"], "tokens": d["usage"]["completion_tokens"],
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "answer": last_boxed(msg.get("content")), "latency": time.time() - t,
            "content": msg.get("content"), "reasoning": msg.get("reasoning_content")}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "bench"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=0, help="client-side cap on in-flight requests (0 = all)")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        async with s.get(base + "/v1/models") as r:
            model = (await r.json())["data"][0]["id"]
        url = base + "/v1/chat/completions"
        if a.mode == "smoke":
            res = await one(s, url, model, PROBLEMS[0][0], a.max_tokens, a.effort)
            print(json.dumps({k: res.get(k) for k in ("finish", "tokens", "prompt_tokens", "answer", "latency")}))
            print("expected 25; content tail:", (res.get("content") or "")[-300:].replace("\n", " "))
            print("reasoning head:", (res.get("reasoning") or "")[:200].replace("\n", " "))
            return
        t0 = time.time()
        sem = asyncio.Semaphore(a.concurrency or a.n)

        async def capped(i):
            async with sem:
                return await one(s, url, model, PROBLEMS[i % len(PROBLEMS)][0], a.max_tokens, a.effort)
        jobs = [capped(i) for i in range(a.n)]
        res = await asyncio.gather(*jobs)
        wall = time.time() - t0
    ok = [r for r in res if "tokens" in r]
    toks = [r["tokens"] for r in ok]
    correct = sum(r["answer"] == PROBLEMS[i % len(PROBLEMS)][1] for i, r in enumerate(res) if "tokens" in r)
    summ = {"model": model, "n": a.n, "ok": len(ok), "errors": len(res) - len(ok), "max_tokens": a.max_tokens,
            "effort": a.effort, "concurrency": a.concurrency or a.n, "wall_s": round(wall, 1), "total_out_tokens": sum(toks),
            "out_tok_per_s": round(sum(toks) / wall, 1), "mean_len": round(statistics.mean(toks), 1) if toks else 0,
            "truncated": sum(r["finish"] == "length" for r in ok), "correct": correct}
    print(json.dumps(summ))
    if a.out:
        with open(a.out, "w") as f:
            f.write(json.dumps(summ) + "\n")
            for r in res:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("content", "reasoning")}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
