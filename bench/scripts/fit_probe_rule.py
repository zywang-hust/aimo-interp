"""Package rule with the white-box length probe: gpt-oss medium cut at T, unfinished samples scored by the probe in
submission/fragility/probe.npz (exactly what the package applies), score = mean est. log length + c*w.
Compared with the capped log length at the same and longer T. Official rows; n of the 16 train36 traces per problem
drawn `reps` times (n=16 has no draw noise, so it is optimistic). Prints the pooled threshold for decision.json.
With --twin: adds c_t * twin (the problem is a non-verbatim copy of a public contest problem: best corpus match
similarity < 0.9995, as the package's describe() computes it; bench/train36_refmatch.jsonl) for a grid of c_t.
  python3 -I bench/scripts/fit_probe_rule.py [--twin]
  python3 -I bench/scripts/fit_probe_rule.py --fit <probe|cap> <n> <T> <c_t>    # one config (threshold for decision.json)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import auc  # noqa: E402
from evalbench import load, lopo_predict  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"


def probe_pred(feats):
    m = np.load(f"{ROOT}/submission/fragility/probe.npz")
    return (((feats - m["mu"]) / m["sd"] - m["pmean"]) @ m["comps"].T) @ m["coef"] + m["b"], int(m["T"]), int(m["layer"])


def main(c=0.1, reps=200, twin_grid=None, one=None):
    z = np.load(f"{ROOT}/bench/runs/trace/feats/train36_medium.npz")
    meta = json.loads(str(z["meta"]))
    ids, ln = np.array([m["id"] for m in meta]), np.array([m["len"] for m in meta], float)
    _, T, L = probe_pred(np.zeros((1, 2880), np.float32))
    pred, _, _ = probe_pred(z["H"][:, list(z["ckpt"]).index(T), list(z["layers"]).index(L)].astype(np.float32))
    D, _ = load()
    D = D[D.pid.isin(set(ids))].reset_index(drop=True)
    y, h, w = D.S.values, D.hard.values, D.w.values
    by = {p: np.where(ids == p)[0] for p in np.unique(ids)}
    sim = {r["pid"]: r["sim"] for r in map(json.loads, open(f"{ROOT}/bench/train36_refmatch.jsonl"))}
    twin = ((D.pid.map(sim).values > 0.6) & (D.pid.map(sim).values < 0.9995)).astype(float)
    rng = np.random.default_rng(0)

    def score(kind, n, TT, ct=0.0):
        s = {}
        for p, ix in by.items():
            ix = rng.choice(ix, n, replace=False)
            if kind == "probe":
                s[p] = np.where(ln[ix] <= TT, np.log(ln[ix]), np.maximum(pred[ix], np.log(TT))).mean()
            else:
                s[p] = np.log(np.minimum(ln[ix], TT)).mean()
        return D.pid.map(s).values + c * w + ct * twin

    print(f"{len(D)} rows / {D.pid.nunique()} problems, hard {h.sum()} rows; probe T={T} L={L}; c={c}, reps={reps}")
    grid = [(k, n, TT, 0.0) for k, n, TT in [("probe", 8, T), ("probe", 12, T), ("probe", 16, T), ("cap", 8, T),
                                              ("cap", 16, T), ("cap", 8, 3584), ("cap", 12, 3584), ("cap", 16, 3584),
                                              ("cap", 16, 4096)]]
    if one:
        grid = [one]
    elif twin_grid:
        print(f"twin rows {int(twin.sum())} (hard {int(twin[h].sum())}, S {int(y[twin > 0].sum())})")
        grid = [("probe", 12, T, ct) for ct in twin_grid] + [("cap", 8, 3584, ct) for ct in twin_grid]
    for kind, n, TT, ct in grid:
        S, acc, au = [], [], []
        for _ in range(reps if n < 16 else 1):
            s = score(kind, n, TT, ct)
            S.append(s)
            pr = lopo_predict(s, D)
            acc.append(((pr == y).mean(), (pr[h] == y[h]).mean()))
            au.append(auc(s[h], ~y[h]))
        S = np.concatenate(S)
        Y, H = np.tile(y, len(acc)), np.tile(h, len(acc))
        cand = np.quantile(S, np.linspace(0.02, 0.98, 400))
        bal = [0.5 * ((S[Y] > t).mean() + (S[~Y] <= t).mean()) for t in cand]
        t = float(cand[int(np.argmax(bal))])
        a = np.mean(acc, 0)
        print(f"{kind:5s} n={n:2d} T={TT:4d} twin {ct:.2f}: hard AUC {np.mean(au):.3f} | LOPO acc all {a[0]:.3f} hard {a[1]:.3f} | "
              f"thr {t:.4f} (pooled bal {max(bal):.3f}, hard acc at thr {((S[H] > t) == Y[H]).mean():.3f})")


if __name__ == "__main__":
    if "--fit" in sys.argv:
        a = sys.argv[sys.argv.index("--fit") + 1:]
        main(one=(a[0], int(a[1]), int(a[2]), float(a[3])))
    else:
        main(twin_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.6] if "--twin" in sys.argv else None)
