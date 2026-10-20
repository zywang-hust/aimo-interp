"""Numeric-counterfactual drop on the bench: p(parent) - p(twin) per config, from pool / twins samples.

A twin is the parent problem with its numbers changed and the answer recomputed, i.e. one re-instantiation of the
problem's symbolic form, which is how the organisers build counterfactual variants. The drop is a direct, dense
robustness target. Output: bench/runs/twin_drop.csv (twin, parent, cfg, p_parent, p_twin, n_parent, n_twin, drop).
  python3 -I bench/scripts/twin_drop.py
"""
import glob
import json
from collections import defaultdict

import pandas as pd

ROOT = "/user/wangzhiyao/projects/interp"


def acc(run, answers):
    """cfg -> id -> (correct, n) over all output files of a run (forward + .rev)."""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    seen = set()
    for f in glob.glob(f"{ROOT}/bench/runs/{run}/out/*.jsonl"):
        cfg = f.split("/")[-1].removesuffix(".jsonl").removesuffix(".rev")
        for line in open(f):
            r = json.loads(line)
            key = (cfg, r["id"], r["k"])
            if key in seen or r["id"] not in answers:
                continue
            seen.add(key)
            c = out[cfg][r["id"]]
            c[0] += str(r.get("answer")) == answers[r["id"]]
            c[1] += 1
    return out


def main():
    ans = {r["id"]: str(r["answer"]) for r in map(json.loads, open(f"{ROOT}/bench/pool/answers.jsonl"))}
    twins = [json.loads(l) for l in open(f"{ROOT}/bench/pool/twins_verified.jsonl")]
    base, twin = acc("pool", ans), acc("twins", ans)
    rows = []
    for cfg in twin:
        for t in twins:
            ct, cp = twin[cfg].get(t["id"]), base[cfg].get(t["parent"])
            if ct and cp and ct[1] >= 4 and cp[1] >= 4:
                rows.append(dict(twin=t["id"], parent=t["parent"], cfg=cfg, p_parent=cp[0] / cp[1], p_twin=ct[0] / ct[1],
                                 n_parent=cp[1], n_twin=ct[1]))
    D = pd.DataFrame(rows)
    D["drop"] = D.p_parent - D.p_twin
    D.to_csv(f"{ROOT}/bench/runs/twin_drop.csv", index=False)
    D["S"] = D["drop"] >= 0.25
    D["R"] = D["drop"] <= 0.10
    print(D.groupby("cfg").agg(n=("drop", "size"), mean_drop=("drop", "mean"), S=("S", "mean"), R=("R", "mean"),
                               p_parent=("p_parent", "mean"), p_twin=("p_twin", "mean")).round(3))
    D["pbin"] = pd.cut(D.p_parent, [-0.01, 0.25, 0.5, 0.75, 0.9, 1.0])
    print(D.groupby("pbin", observed=True).agg(n=("drop", "size"), mean_drop=("drop", "mean"), S=("S", "mean")).round(3))


if __name__ == "__main__":
    main()
