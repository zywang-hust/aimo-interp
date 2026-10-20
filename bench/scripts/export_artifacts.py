"""Export length-probe artifacts for the submission (submission/fragility/artifacts/<cfg>.npz).

Per config: refit the ridge on all bench problems at the layer chosen by train_probe.py (from its official-
transfer CSV), and store everything inference needs:
  layer, mu, sd, w, ym       standardize -> linear -> predicted log mean completion length
  ref                        sorted out-of-fold bench predictions of problems with p_base >= --p-min
  thr_pct                    Spurious iff the prediction's percentile within ref is > thr_pct
Run train_probe.py for every config first.
  python3 -I bench/scripts/export_artifacts.py [--p-min 0.75] [--thr-pct 0.58]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/user/wangzhiyao/projects/interp"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_probe import targets  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--p-min", type=float, default=0.75)
ap.add_argument("--thr-pct", type=float, default=0.58)
ap.add_argument("--out-dir", default=f"{ROOT}/submission/fragility/artifacts")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
ans = {r["id"]: str(r["answer"]) for r in map(json.loads, open(f"{ROOT}/bench/pool/answers.jsonl"))}

for f in sorted(glob.glob(f"{ROOT}/bench/runs/probe/*.loglen.last.csv")):
    cfg = os.path.basename(f).split(".loglen")[0]
    meta = pd.read_csv(f)
    L = int(meta.layer.iloc[0])
    ids = pd.read_csv(f"{ROOT}/bench/runs/probe/{cfg}.bench_ids.csv")
    cv = np.load(f"{ROOT}/bench/runs/probe/{cfg}.loglen.last.benchcv.npy")
    # rebuild X, y in the same order as train_probe.py wrote bench_ids
    feats = {}
    for d in ["pool", "twins"]:
        p = f"{ROOT}/bench/feats/{d}/{cfg}.npz"
        if os.path.exists(p):
            F = np.load(p)
            feats.update(zip(F["ids"], F["last"][:, L].astype(np.float32)))
    Y = pd.concat([targets(f"{ROOT}/bench/runs/{s}/out/{cfg}*.jsonl", ans) for s in ["pool", "twins"]
                   if glob.glob(f"{ROOT}/bench/runs/{s}/out/{cfg}*.jsonl")])
    Y = Y[~Y.index.duplicated()]
    X = np.stack([feats[i] for i in ids.id])
    y = Y.loc[ids.id, "loglen"].values
    # lambda: train_probe picks it per layer by CV; refit with the same grid choice via a small inner CV
    from train_probe import cv_pred, spearman  # noqa: E402
    parent = {r["id"]: r.get("parent", r["id"]) for r in map(json.loads, open(f"{ROOT}/bench/pool/answers.jsonl"))}
    groups = np.array([parent[i] for i in ids.id])
    lam = max([0.01, 0.1, 1, 10, 100], key=lambda l: spearman(cv_pred(X, y, l, groups), y))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = y.mean()
    a = np.linalg.solve(Z @ Z.T + lam * len(y) * np.eye(len(y)), y - ym)
    w = Z.T @ a
    ref = np.sort(cv[ids.p.values >= args.p_min])
    np.savez(f"{args.out_dir}/{cfg}.npz", layer=L, mu=mu, sd=sd, w=w, ym=ym, ref=ref, thr_pct=args.thr_pct, lam=lam)
    print(f"{cfg}: layer {L} lam {lam} n_train {len(y)} n_ref {len(ref)} d {X.shape[1]}")
