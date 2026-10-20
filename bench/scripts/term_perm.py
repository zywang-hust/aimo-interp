"""Permutation tests for problem-level terms added to the length score (NOTES §2.1, §5). Terms are fixed in advance:
  twin     0.6 < corpus similarity < 0.9995 (the package's is_twin)
  new      source is AIME 2026 or AIMO3 (vs AIME 2025 / AIMO2), from train36_refmatch
  textlen  log characters of the statement, z-scored over the 36 problems
Base score = mean log length of the 16 train36all gpt-oss medium samples capped at 3584 + 0.1w. The gain is the hard-
region leave-one-problem-out accuracy of base + c*term minus that of base, best over c in ±(0.2, 0.4, 0.6) (twin: +
only, as submitted), so picking c and its sign is part of the test. Null: the term's values shuffled across problems.
  python3 -I bench/scripts/term_perm.py [n_perm]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalbench import load, lopo_predict  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"
D, runs = load()
ref = {r["pid"]: r for r in map(json.loads, open(f"{ROOT}/bench/train36_refmatch.jsonl"))}
txt = {r["pid"]: len(r["problem"]) for r in map(json.loads, open(f"{ROOT}/bench/train36_problems.jsonl"))}
y, h, w = D.S.values, D.hard.values, D.w.values
pids = D.pid.unique()
base = D.pid.map({p: np.log(np.minimum(v, 3584)).mean() for p, v in runs["medium"].items()}).values + 0.1 * w
a0 = (lopo_predict(base, D)[h] == y[h]).mean()

tl = np.log([txt[p] for p in ref])
TERMS = {
    "twin": ({p: float(0.6 < ref[p]["sim"] < 0.9995) for p in pids}, (0.2, 0.4, 0.6)),
    "new": ({p: float(ref[p]["src"] in ("aime_2026", "aimo3")) for p in pids}, (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6)),
    "textlen": ({p: (np.log(txt[p]) - tl.mean()) / tl.std() for p in pids}, (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6)),
}


def gain(vals, cs):
    t = D.pid.map(vals).values
    accs = [(lopo_predict(base + c * t, D)[h] == y[h]).mean() for c in cs]
    return max(accs) - a0, cs[int(np.argmax(accs))]


n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(0)
print(f"hard rows {h.sum()}, problems {len(np.unique(D.pid[h]))}; base hard LOPO acc {a0:.3f}")
for name, (vals, cs) in TERMS.items():
    g, c = gain(vals, cs)
    v = np.array([vals[p] for p in pids])
    null = np.array([gain(dict(zip(pids, rng.permutation(v))), cs)[0] for _ in range(n_perm)])
    hp = np.unique(D.pid[h])
    s = np.array([y[(D.pid == p).values].mean() > 0.5 for p in hp])
    tv = np.array([vals[p] for p in hp])
    print(f"{name:8s} gain {g:+.3f} (c={c:+.1f}); null mean {null.mean():+.3f} q95 {np.quantile(null, .95):+.3f}; "
          f"p = {np.mean(null >= g - 1e-12):.3f} | hard term mean S {tv[s].mean():+.2f} R {tv[~s].mean():+.2f}")
