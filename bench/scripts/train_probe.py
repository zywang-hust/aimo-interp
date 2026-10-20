"""Length / accuracy probe: prefill hidden states -> log mean completion length (and p_base) of a model.

Train on bench pool problems (features: extract_feats.py on the pool; targets: pool samples), cross-validated
by problem, then transfer to the official train-main-v2 problems and score the prediction as a Spurious
detector against the official labels (the real test: no official label is used for training).

  python3 -I bench/scripts/train_probe.py --model-key qwen3.5-4b [--target loglen|p] [--feat last|mean]

Ridge regression in dual form on standardized features, lambda picked by inner CV; reports per-layer CV
Spearman, then the best layer's transfer AUC (and the AUC of the true measured length for reference).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/user/wangzhiyao/projects/interp"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_KEYS, auc  # noqa: E402
from common import load_samples as load  # noqa: E402


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = y.mean()
    K = Z @ Z.T
    a = np.linalg.solve(K + lam * len(y) * np.eye(len(y)), y - ym)
    w = Z.T @ a
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def cv_pred(X, y, lam, groups, k=5, seed=0):
    """k-fold CV with whole groups held out (a twin always shares the fold of its parent problem)."""
    ug = np.random.default_rng(seed).permutation(np.unique(groups))
    pred = np.zeros(len(y))
    for f in range(k):
        te = np.where(np.isin(groups, ug[f::k]))[0]
        tr = np.where(~np.isin(groups, ug[f::k]))[0]
        pred[te] = ridge_fit(X[tr], y[tr], lam)(X[te])
    return pred


def targets(pattern, answers):
    S = load(pattern)
    S["ok"] = [a is not None and str(a) == answers.get(i) for a, i in zip(S.answer, S.id)]
    g = S.groupby("id").agg(p=("ok", "mean"), len=("completion_tokens", "mean"), n=("ok", "size"))
    g["loglen"] = np.log(g["len"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--effort", default="default")
    ap.add_argument("--target", default="loglen", choices=["loglen", "p"])
    ap.add_argument("--feat", default="last", choices=["last", "mean"])
    ap.add_argument("--lams", default="0.01,0.1,1,10,100")
    ap.add_argument("--min-n", type=int, default=4)
    ap.add_argument("--twins", action="store_true", help="also train on the verified twins (twins)")
    args = ap.parse_args()
    cfg = f"{args.model_key}__{args.effort}"

    pool_ans = {r["id"]: str(r["answer"]) for r in map(json.loads, open(f"{ROOT}/bench/pool/answers.jsonl"))}
    parent = {r["id"]: r.get("parent", r["id"]) for r in map(json.loads, open(f"{ROOT}/bench/pool/answers.jsonl"))}
    parts = [("pool", "pool")] + ([("twins", "twins")] if args.twins else [])
    Xs, ys, gs, bench_ids, bench_p = [], [], [], [], []
    for step, fdir in parts:
        Y = targets(f"{ROOT}/bench/runs/{step}/out/{cfg}*.jsonl", pool_ans)  # incl. reverse-helper files
        Y = Y[Y.n >= args.min_n]
        F = np.load(f"{ROOT}/bench/feats/{fdir}/{cfg}.npz")
        ids = list(F["ids"])
        keep = [i for i, x in enumerate(ids) if x in Y.index]
        Xs.append(F[args.feat][keep].astype(np.float32))  # [N, L+1, d]
        ys.append(Y.loc[[ids[i] for i in keep], args.target].values)
        gs += [parent[ids[i]] for i in keep]
        bench_ids += [ids[i] for i in keep]
        bench_p += list(Y.loc[[ids[i] for i in keep], "p"].values)
        print(f"{cfg} {fdir}: {len(keep)} problems with targets; mean len {Y['len'].mean():.0f}, mean p {Y.p.mean():.3f}")
    X_all, y, groups = np.concatenate(Xs), np.concatenate(ys), np.array(gs)
    print(f"target={args.target} feat={args.feat}; {len(y)} problems in {len(np.unique(groups))} CV groups")

    lams = [float(x) for x in args.lams.split(",")]
    res = []
    for L in range(X_all.shape[1]):
        best = max((spearman(cv_pred(X_all[:, L], y, lam, groups), y), lam) for lam in lams)
        res.append((L, *best))
    res = pd.DataFrame(res, columns=["layer", "cv_spearman", "lam"])
    print(res.round(3).to_string(index=False, max_rows=60))
    L, _, lam = res.loc[res.cv_spearman.idxmax()]
    L = int(L)

    # transfer to the official problems
    T = pd.read_json(f"{ROOT}/bench/train_main_v2_rows.jsonl", lines=True)
    T = T[(T.model_id == MODEL_KEYS[args.model_key]) & (T.reasoning_effort == args.effort)]
    G = np.load(f"{ROOT}/bench/feats/train36/{cfg}.npz")
    pred = dict(zip(G["ids"], ridge_fit(X_all[:, L], y, lam)(G[args.feat][:, L].astype(np.float32))))
    T = T.assign(pred=T.pid.map(pred))
    lab = T[T.is_robust.notna()]
    sign = 1 if args.target == "loglen" else -1  # long / low-p -> Spurious
    ref_ans = {r["id"]: str(r["answer"]) for r in map(json.loads, open(f"{ROOT}/bench/train36_answers.jsonl"))}
    M = targets(f"{ROOT}/bench/runs/step1/out/{cfg}.jsonl", ref_ans)
    lab = lab.assign(measured=lab.pid.map(M[args.target]))
    print(f"\nbest layer {L} (lam {lam}); transfer to {len(lab)} labelled official rows of {cfg}:")
    print(f"  AUC(Spurious | probe prediction)   = {auc(sign * lab.pred, lab.is_robust == 0):.3f}")
    print(f"  AUC(Spurious | measured {args.target:6s})    = {auc(sign * lab.measured, lab.is_robust == 0):.3f}")
    os.makedirs(f"{ROOT}/bench/runs/probe", exist_ok=True)
    # out-of-fold bench predictions at the chosen layer: reference distribution for calibrating official scores
    np.save(f"{ROOT}/bench/runs/probe/{cfg}.{args.target}.{args.feat}.benchcv.npy", cv_pred(X_all[:, L], y, lam, groups))
    if args.target == "loglen":  # export_artifacts pairs this file with the loglen benchcv
        pd.DataFrame({"id": bench_ids, "p": bench_p}).to_csv(f"{ROOT}/bench/runs/probe/{cfg}.bench_ids.csv", index=False)
    T.assign(measured=T.pid.map(M[args.target]), layer=L, target=args.target).to_csv(
        f"{ROOT}/bench/runs/probe/{cfg}.{args.target}.{args.feat}.csv", index=False)
    print(f"  spearman(prediction, measured) on all {len(T)} official problems of this config = "
          f"{spearman(T.pred, T.pid.map(M[args.target])):.3f}")


if __name__ == "__main__":
    main()
