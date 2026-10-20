"""Honest local bench for decision rules on the official labelled rows.

Rules are scored the way the package would score them on the eval box:
  - rollout features are recomputed from n random samples of train36all (n=16 per problem, sglang, same prompt; plus
    train36x for medium64, 64 per problem, T < 4096), capped at T tokens, so the eval-time sampling noise is included (R repeats);
  - thresholds are fitted leave-one-problem-out (the held-out problem's rows never see their own labels);
  - uncertainty: bootstrap over problems (labels are a property of the problem: 32 of 33 problems have one label).
Reported on all rows and on the hard rows (qwen length z > -0.42, the validation-like regime).
  python3 -I bench/scripts/evalbench.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import auc  # noqa: E402
from val_like import table  # noqa: E402
from val_median import weakness  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"


def load():
    D = table()
    D["w"] = D.cfg.map(weakness())
    D["S"] = ~D.is_robust.astype(bool)
    D["hard"] = D["z:qwen3.5-4b"] > -0.42
    D["oss"] = D.cfg.str.startswith("gpt-oss")
    runs = {}
    for e in ("low", "medium", "high"):
        R = pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/bench/runs/train36all/out/gpt-oss-120b__{e}.jsonl")])
        runs[e] = {k: g.completion_tokens.values for k, g in R.groupby("id")}
    # medium64: train36all's 16 plus train36x's 48 capped at 4096 (generation stopped there), so valid for T < 4096
    X = pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/bench/runs/train36x/out/gpt-oss-120b__medium.jsonl")])
    runs["medium64"] = {k: np.concatenate([v, X.completion_tokens.values[X.id == k]]) for k, v in runs["medium"].items()}
    return D.reset_index(drop=True), runs


def rollout_feat(runs, pids, rng, n, T, kind):
    out = {}
    for k, v in runs.items():
        s = rng.choice(v, n, replace=False)
        out[k] = np.mean(s > T) if kind == "frac" else np.log(np.minimum(s, T)).mean()
    return np.array([out[p] for p in pids])


def lopo_predict(score, D, fixed_thr=None):
    """Spurious if score > thr; thr maximises balanced accuracy on the other problems (or is fixed)."""
    if fixed_thr is not None:
        return score > fixed_thr
    pred = np.zeros(len(D), bool)
    y = D.S.values
    cands = np.unique(score)
    for p in D.pid.unique():
        m = (D.pid == p).values
        tr = ~m
        best, bt = -1, None
        for t in cands:
            sp = score[tr] > t
            b = 0.5 * (sp[y[tr]].mean() + (~sp[~y[tr]]).mean())
            if b > best:
                best, bt = b, t
        pred[m] = score[m] > bt
    return pred


def lopo_logit(X, D, C=1.0):
    """Logistic regression fitted leave-one-problem-out, class-balanced; returns P(Spurious) per row."""
    from sklearn.linear_model import LogisticRegression
    out = np.zeros(len(D))
    y = D.S.values
    for p in D.pid.unique():
        m = (D.pid == p).values
        mu, sd = X[~m].mean(0), X[~m].std(0) + 1e-9
        clf = LogisticRegression(C=C, class_weight="balanced").fit((X[~m] - mu) / sd, y[~m])
        out[m] = clf.predict_proba((X[m] - mu) / sd)[:, 1]
    return out


def rules(D, runs, rng):
    """name -> (score array, fixed threshold or None). One draw of eval-time sampling noise."""
    w = D.w.values
    sky = D["z:skywork-or1-7b"].values
    r = {}
    f43 = rollout_feat(runs["medium"], D.pid, rng, 4, 3072, "frac")
    r["C0 frac n4 T3072 +.15w (submitted, thr .55)"] = (f43 + 0.15 * w, 0.55)
    r["C0' same, LOPO thr"] = (f43 + 0.15 * w, None)
    l84 = rollout_feat(runs["medium"], D.pid, rng, 8, 4096, "loglen")
    r["C1 loglen n8 T4096 +.1w"] = (l84 + 0.1 * w, None)
    r["C1' loglen n8 T4096 (no w)"] = (l84, None)
    l166 = rollout_feat(runs["medium"], D.pid, rng, 16, 6144, "loglen")
    r["C1'' loglen n16 T6144 +.1w"] = (l166 + 0.1 * w, None)
    zl = (l84 - l84.mean()) / l84.std()
    zs = (sky - sky.mean()) / sky.std()
    r["C2 oss rows: C1 ; small rows: skywork"] = (np.where(D.oss, zl + 0.25 * w, zs), None)
    r["C3 skywork reader +.5w"] = (zs + 0.5 * w, None)
    high = (D.cfg == "gpt-oss-120b__high").values.astype(float)
    r["L1 logit(len, w)"] = (lopo_logit(np.c_[l84, w], D), 0.5)
    r["L2 logit(len, is_high)"] = (lopo_logit(np.c_[l84, high], D), 0.5)
    r["L3 logit(len, w, is_high)"] = (lopo_logit(np.c_[l84, w, high], D), 0.5)
    r["L4 logit(len, is_high, sky)"] = (lopo_logit(np.c_[l84, high, sky], D), 0.5)
    return r


def main():
    D, runs = load()
    y = D.S.values
    hard = D.hard.values
    rng = np.random.default_rng(0)
    R = 30
    acc = {}
    for _ in range(R):
        for name, (s, thr) in rules(D, runs, rng).items():
            pred = lopo_predict(s, D, thr)
            acc.setdefault(name, []).append((pred == y, auc(s, ~y), auc(s[hard], ~y[hard])))
    pids = D.pid.unique()
    boot = np.random.default_rng(1)
    B = [boot.choice(pids, len(pids)) for _ in range(500)]
    idx = {p: np.where(D.pid.values == p)[0] for p in pids}
    print(f"{len(D)} rows, {len(pids)} problems; hard {hard.sum()} rows / {D[hard].pid.nunique()} problems; R={R}")
    print(f"{'rule':44s} {'AUC all':>8s} {'AUC hard':>9s} {'acc all':>8s} {'acc hard [90% CI]':>22s}")
    for name, v in acc.items():
        ok = np.mean([a[0] for a in v], 0)  # per-row accuracy averaged over sampling draws
        a_all, a_hard = np.mean([a[1] for a in v]), np.mean([a[2] for a in v])
        hb = []
        for b in B:
            rows = np.concatenate([idx[p] for p in b])
            rows = rows[hard[rows]]
            if len(rows):
                hb.append(ok[rows].mean())
        lo, hi = np.quantile(hb, [0.05, 0.95])
        print(f"{name:44s} {a_all:8.3f} {a_hard:9.3f} {ok.mean():8.3f} {ok[hard].mean():8.3f} [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
