"""Fit the white-box length probe on bench traces (official problems removed) and write it for the package.

For each checkpoint T and layer: ridge on PCA(k) of the standardised hidden state (mean of the last 16 generated
tokens at T), trained on traces still running at T, target log final length. Model choice uses grouped 5-fold CV R²
on bench only (no labels). Writes submission/fragility/probe.npz for the chosen (T, layer) [or the one given].
  python3 -I bench/scripts/fit_len_probe.py <bench shards,comma-separated> [T L]
"""
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from len_probe import read  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"
official = {json.loads(l)["id"] for l in open(f"{ROOT}/bench/train36_input.jsonl")}


def fit(X, y, k, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    p = PCA(k).fit((X - mu) / sd)
    r = Ridge(alpha).fit(p.transform((X - mu) / sd), y)
    return {"mu": mu, "sd": sd, "pmean": p.mean_, "comps": p.components_, "coef": r.coef_, "b": r.intercept_}


def apply(m, X):
    return (((X - m["mu"]) / m["sd"] - m["pmean"]) @ m["comps"].T) @ m["coef"] + m["b"]


B = read(sys.argv[1])
keep = ~np.isin(B["ids"], list(official))
print(f"bench traces {keep.sum()} of {len(keep)} ({len(set(B['ids'][keep]))} problems)")
best = None
for c, T in enumerate(B["ck"]):
    alive = keep & (B["len"] > T)
    y, g = np.log(B["len"][alive]), B["ids"][alive]
    for li, L in enumerate(B["ly"]):
        X = B["H"][alive, c, li]
        for k, alpha in [(32, 10.0), (64, 10.0), (128, 30.0)]:
            pred = np.zeros_like(y)
            for tr, te in GroupKFold(5).split(X, y, g):
                pred[te] = apply(fit(X[tr], y[tr], k, alpha), X[te])
            r2 = 1 - np.mean((pred - y) ** 2) / np.var(y)
            print(f"T={T:5d} L={L:2d} k={k:3d}: alive {alive.sum():5d} CV R2 {r2:.3f} rmse {np.sqrt(np.mean((pred - y) ** 2)):.3f}",
                  flush=True)
            if len(sys.argv) > 3 and (T, L) == (int(sys.argv[2]), int(sys.argv[3])) and (best is None or r2 > best[0]):
                best = (r2, c, li, k, alpha, T, L)
if best:
    r2, c, li, k, alpha, T, L = best
    alive = keep & (B["len"] > T)
    m = fit(B["H"][alive, c, li], np.log(B["len"][alive]), k, alpha)
    out = f"{ROOT}/submission/fragility/probe.npz"
    np.savez(out, **{kk: np.asarray(v, np.float32) for kk, v in m.items()}, T=T, layer=L, k=k, cv_r2=r2)
    print(f"wrote {out}: T={T} L={L} k={k} CV R2 {r2:.3f}")
