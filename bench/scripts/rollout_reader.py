"""Rollout readers: one config's measured behaviour on every official problem, used as a problem-level score
for all rows (labels are a property of the problem). Data: bench/runs/train36all (n=16 per problem and config).
Reports AUC on all rows / hard rows (qwen length z > -0.42, the validation-like regime) / fresh problems.
  python3 -I bench/scripts/rollout_reader.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import auc  # noqa: E402
from val_like import table  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"


def measured():
    ans = {r["id"]: str(r["answer"]) for r in map(json.loads, open(f"{ROOT}/bench/train36_answers.jsonl"))}
    out = {}
    for f in glob.glob(f"{ROOT}/bench/runs/train36all/out/*.jsonl"):
        cfg = os.path.basename(f).removesuffix(".jsonl")
        R = pd.DataFrame([json.loads(l) for l in open(f)])
        if R.empty:
            continue
        R["ok"] = R.answer.astype(str) == R.id.map(ans)
        R["nan"] = R.answer.astype(str) == "NaN"
        R["trunc"] = R.finish == "length"
        g = R.groupby("id").agg(p=("ok", "mean"), loglen=("completion_tokens", lambda x: np.log(x).mean()),
                                sdlen=("completion_tokens", lambda x: np.log(x).std()),
                                agree=("answer", lambda x: x.astype(str).value_counts(normalize=True, dropna=False).max()),
                                trunc=("trunc", "mean"), n=("ok", "size"))
        out[cfg] = g
    return out


def main():
    A = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/bench/train36_answers.jsonl"))}
    D = table()
    y = D.is_robust.values.astype(bool)
    hard = D["z:qwen3.5-4b"].values > -0.42
    fresh = D.pid.map(lambda p: A[p]["src"] in ("aime_2026", "aimo3")).values
    for cfg, g in sorted(measured().items()):
        if g.n.min() < 8:
            print(f"{cfg}: incomplete ({len(g)} problems, min n {g.n.min()})")
            continue
        for feat, sign in [("loglen", 1), ("1-p", -1), ("1-agree", -1), ("sdlen", 1)]:
            col = {"1-p": "p", "1-agree": "agree"}.get(feat, feat)
            v = sign * D.pid.map(g[col]).values.astype(float)
            print(f"{cfg:26s} {feat:8s} all {auc(v, y):.3f}  hard {auc(v[hard], y[hard]):.3f}  "
                  f"fresh {auc(v[fresh], y[fresh]):.3f}  fresh&hard {auc(v[fresh & hard], y[fresh & hard]):.3f}")


if __name__ == "__main__":
    main()
