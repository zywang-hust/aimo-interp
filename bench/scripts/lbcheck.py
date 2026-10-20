"""Does the bench predict the public leaderboard? Every submitted rule is replayed on the official rows with the
threshold it was submitted with and eval-time sampling noise (n samples drawn from the 64 gpt-oss medium samples per
problem, capped at T; the probe rule draws from the 16 with hidden states).
The public validation set is 21 cases from 14 problems, about half Spurious, almost all in the hard region; a
validation draw here is 7 Spurious + 7 Robust hard problems (with replacement) and 21 rows over them (each problem
at least once). Prints the mean and 90% interval of the simulated LB score next to the real one, with the submitted
threshold and with the threshold refitted leave-one-problem-out (honest for rules fitted on these rows).
  python3 -I bench/scripts/lbcheck.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalbench import load, lopo_predict, rollout_feat  # noqa: E402
from fit_probe_rule import probe_pred  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"


def probe_feat(by, ln, pred, pids, rng, n, T):
    s = {}
    for p, ix in by.items():
        ix = rng.choice(ix, n, replace=False)
        s[p] = np.where(ln[ix] <= T, np.log(ln[ix]), np.maximum(pred[ix], np.log(T))).mean()
    return np.array([s[p] for p in pids])


def sim(ok, y, rng, S_p, R_p, rows, vals):
    out = []
    for _ in range(vals):
        probs = list(rng.choice(S_p, 7)) + list(rng.choice(R_p, 7))
        pick = probs + list(rng.choice(probs, 21 - len(probs)))
        out.append(np.mean([ok[rng.choice(rows[p])] for p in pick]))
    return out


def main(draws=40, vals=200):
    D, runs = load()
    z = np.load(f"{ROOT}/bench/runs/trace/feats/train36_medium.npz")
    meta = json.loads(str(z["meta"]))
    ids, ln = np.array([m["id"] for m in meta]), np.array([m["len"] for m in meta], float)
    _, T, L = probe_pred(np.zeros((1, 2880), np.float32))
    pred, _, _ = probe_pred(z["H"][:, list(z["ckpt"]).index(T), list(z["layers"]).index(L)].astype(np.float32))
    by = {p: np.where(ids == p)[0] for p in np.unique(ids)}
    D = D[D.pid.isin(set(ids))].reset_index(drop=True)
    refsim = {r["pid"]: r["sim"] for r in map(json.loads, open(f"{ROOT}/bench/train36_refmatch.jsonl"))}
    twin = ((D.pid.map(refsim) > 0.6) & (D.pid.map(refsim) < 0.9995)).values.astype(float)
    y, h, w, pids = D.S.values, D.hard.values, D.w.values, D.pid.values
    qz = D["z:qwen3.5-4b"].values
    m = runs["medium64"]  # all rules here use T <= 4096
    rules = {  # id: (score function of rng, threshold, real LB score)
        "1000189 qwen reader": (lambda r: qz, -0.404, 0.571),
        "1000361 frac n4 T3072 +.15w": (lambda r: rollout_feat(m, pids, r, 4, 3072, "frac") + 0.15 * w, 0.55, 0.714),
        "1000540 loglen n8 T4096 +.1w": (lambda r: rollout_feat(m, pids, r, 8, 4096, "loglen") + 0.1 * w, 8.07, 0.667),
        "1000708* loglen n8 T3584 +.1w (no d)": (lambda r: rollout_feat(m, pids, r, 8, 3584, "loglen") + 0.1 * w, 8.00, 0.762),
        "1003752 probe n12 T1536 +.1w": (lambda r: probe_feat(by, ln, pred, pids, r, 12, 1536) + 0.1 * w, 8.3144, 0.762),
        "1004127 same (rerun, other host?)": (lambda r: probe_feat(by, ln, pred, pids, r, 12, 1536) + 0.1 * w, 8.3144, 0.619),
        "1004192 same, v2 engine": (lambda r: probe_feat(by, ln, pred, pids, r, 12, 1536) + 0.1 * w, 8.3144, 0.714),
        "1003349 twin n12 T3072 +.1w +.4t": (lambda r: rollout_feat(m, pids, r, 12, 3072, "loglen") + 0.1 * w + 0.4 * twin,
                                              7.9501, 0.667),
        "1003501 same, v2 engine": (lambda r: rollout_feat(m, pids, r, 12, 3072, "loglen") + 0.1 * w + 0.4 * twin,
                                     7.9501, 0.714),
    }
    hp = np.unique(pids[h])
    S_p = [p for p in hp if y[pids == p].mean() > 0.5]
    R_p = [p for p in hp if p not in S_p]
    rows = {p: np.where((pids == p) & h)[0] for p in hp}
    print(f"hard pool: {len(S_p)} S / {len(R_p)} R problems; {draws} noise draws x {vals} validation draws")
    rng = np.random.default_rng(0)
    print(f"{'rule':38s} {'sim LB [90%]':>20s} {'P(sim<=real)':>12s} {'real':>6s} | {'LOPO thr [90%]':>20s}")
    for name, (f, thr, real) in rules.items():
        sc, sl = [], []
        for _ in range(draws):
            s = f(rng)
            sc += sim((s > thr) == y, y, rng, S_p, R_p, rows, vals)
            sl += sim(lopo_predict(s, D) == y, y, rng, S_p, R_p, rows, vals)
        sc, sl = np.array(sc), np.array(sl)
        q, ql = np.quantile(sc, [0.05, 0.95]), np.quantile(sl, [0.05, 0.95])
        pr = f"{np.mean(sc <= real + 1e-3):.2f}" if real is not None else "-"
        print(f"{name:38s} {sc.mean():.3f} [{q[0]:.2f}, {q[1]:.2f}] {pr:>12s} {real if real is not None else '-':>6} | "
              f"{sl.mean():.3f} [{ql[0]:.2f}, {ql[1]:.2f}]")

if __name__ == "__main__":
    main()
