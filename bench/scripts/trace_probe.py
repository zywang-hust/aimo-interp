"""Do mid-trace features of gpt-oss-medium rollouts (entropy / logprob / hidden states at checkpoint t) separate
Robust vs Spurious beyond the rollout length the package already uses?

Input: engine/evalrt/trace_feats.py output on the train36all medium traces.
Per problem we average over its traces, join to the official rows, and report
  - AUC (S high) of each scalar on all / hard rows, and partial AUC after regressing out log-length (capped 4096),
  - LOPO logistic on [loglen, w, scalar] vs [loglen, w], hard-row accuracy,
  - hidden-state probes: LOPO logistic on [PCA(k) of layer L at checkpoint t, loglen, w], against a null where the
    hidden states are shuffled across problems (labels, loglen, w kept).
  python3 -I bench/scripts/trace_probe.py bench/runs/trace/feats/train36_medium.npz
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import auc  # noqa: E402
from evalbench import load, lopo_logit, lopo_predict  # noqa: E402

z = np.load(sys.argv[1])
H, E, meta, CK, LY = z["H"].astype(np.float32), z["E"], json.loads(str(z["meta"])), list(z["ckpt"]), list(z["layers"])
ids = np.array([m["id"] for m in meta])
D, runs = load()
D = D[D.pid.isin(set(ids))].reset_index(drop=True)
y, hard, w = D.S.values, D.hard.values, D.w.values
pids = D.pid.values
loglen = np.array([np.log(np.minimum(runs["medium"][p], 4096)).mean() for p in pids])
print(f"{len(D)} rows / {D.pid.nunique()} problems, hard {hard.sum()} rows / {D[hard].pid.nunique()} problems")
print(f"loglen T4096: AUC all {auc(loglen, ~y):.3f} hard {auc(loglen[hard], ~y[hard]):.3f}")


def per_problem(X):
    m = {p: X[ids == p].mean(0) for p in set(ids)}
    return np.stack([m[p] for p in pids])


def resid(a, b):
    A = np.c_[np.ones_like(b), b]
    return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]


base = lopo_predict(lopo_logit(np.c_[loglen, w], D), D, 0.5)
print(f"base LOPO logit(loglen, w): acc all {np.mean(base == y):.3f} hard {np.mean(base[hard] == y[hard]):.3f}")
names = ["ent_mean", "ent_std", "logp_mean"]
print(f"{'feature':22s} {'AUC all':>8s} {'AUC hard':>9s} {'pAUC hard|len':>14s} {'+len,w acc hard':>16s}")
for j, nm in enumerate(names):
    for c, t in enumerate(CK):
        s = per_problem(E[:, c, j])
        r = resid(s, loglen)
        pr = lopo_predict(lopo_logit(np.c_[loglen, w, s], D), D, 0.5)
        print(f"{nm + '@' + str(t):22s} {auc(s, ~y):8.3f} {auc(s[hard], ~y[hard]):9.3f} "
              f"{auc(r[hard], ~y[hard]):14.3f} {np.mean(pr[hard] == y[hard]):16.3f}")


def hid_probe(X, yy, k=4, C=0.1):
    """LOPO: PCA(k) fitted on training problems, logistic on (pca, loglen, w). returns hard acc, hard auc."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    out = np.zeros(len(D))
    for p in np.unique(pids):
        m = pids == p
        pca = PCA(k).fit(X[~m])
        F = np.c_[pca.transform(X), loglen, w]
        mu, sd = F[~m].mean(0), F[~m].std(0) + 1e-9
        clf = LogisticRegression(C=C, class_weight="balanced").fit((F[~m] - mu) / sd, yy[~m])
        out[m] = clf.predict_proba((F[m] - mu) / sd)[:, 1]
    return np.mean((out[hard] > 0.5) == yy[hard]), auc(out[hard], ~yy[hard])


rng = np.random.default_rng(0)
print("\nhidden-state probes (PCA4 + loglen + w, LOPO); null = hidden states permuted across problems (30 draws),"
      " labels / loglen / w kept")
up = np.unique(pids)
b = lopo_logit(np.c_[loglen, w], D)
print(f"no hidden state: hard acc {np.mean((b[hard] > .5) == y[hard]):.3f} AUC {auc(b[hard], ~y[hard]):.3f}")
for c, t in enumerate(CK):
    if t < 1024:
        continue
    for li, L in enumerate(LY):
        X = per_problem(H[:, c, li])
        a, u = hid_probe(X, y)
        null = []
        for _ in range(30):
            perm = dict(zip(up, rng.permutation(up)))
            Xp = np.stack([X[pids == perm[p]][0] for p in pids])
            null.append(hid_probe(Xp, y)[1])
        print(f"t={t:5d} L={L:2d}: hard acc {a:.3f} AUC {u:.3f} | null AUC median {np.median(null):.3f} "
              f"p={np.mean(np.array(null) >= u):.2f}")
