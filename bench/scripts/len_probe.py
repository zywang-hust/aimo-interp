"""White-box "remaining effort" probe: from the residual stream of a truncated gpt-oss-medium rollout at step T,
predict the trace's final log length, and use it in place of the cap for unfinished samples.

Probe: ridge on PCA of a layer's hidden state at T (H = mean of last 16 tokens, M = mean over the whole prefix),
trained on traces still running at T. Training traces come from a problem-disjoint set (bench traces, the official
problems removed), or, without one, leave-one-problem-out on the official traces (target is length, not R/S).
Scoring on the official rows follows evalbench: n of the 16 traces per problem drawn R times, threshold LOPO.
  python3 -I bench/scripts/len_probe.py <official.npz> [bench.npz]
"""
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import auc  # noqa: E402
from evalbench import load, lopo_predict  # noqa: E402


def read(paths):
    """one npz of trace_feats.py, or several shards joined with commas."""
    parts = []
    for path in paths.split(","):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        parts.append({"H": z["H"].astype(np.float32), "M": z["M"].astype(np.float32) if "M" in z else None,
                      "ids": np.array([m["id"] for m in meta]), "len": np.array([m["len"] for m in meta], float),
                      "ck": list(z["ckpt"]), "ly": list(z["layers"])})
    out = dict(parts[0])
    for k in ("H", "M", "ids", "len"):
        if out[k] is not None:
            out[k] = np.concatenate([q[k] for q in parts])
    return out


def fit(X, y, k=128, alpha=30.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    p = PCA(min(k, len(X) - 1)).fit((X - mu) / sd)
    r = Ridge(alpha).fit(p.transform((X - mu) / sd), y)
    return lambda A: r.predict(p.transform((A - mu) / sd))


def probe_preds(O, B, c, li, kind):
    """predicted final log length for every official trace at checkpoint index c (only meaningful if alive)."""
    T = O["ck"][c]
    out = np.full(len(O["ids"]), np.nan)
    if B is not None:
        keep = ~np.isin(B["ids"], np.unique(O["ids"])) & (B["len"] > T)
        f = fit(B[kind][keep, c, li], np.log(B["len"][keep]))
        return f(O[kind][:, c, li])
    for p in np.unique(O["ids"]):
        m = O["ids"] == p
        tr = ~m & (O["len"] > T)
        out[m] = fit(O[kind][tr, c, li], np.log(O["len"][tr]))(O[kind][m, c, li])
    return out


def main():
    O = read(sys.argv[1])
    B = read(sys.argv[2]) if len(sys.argv) > 2 else None
    D, runs = load()
    D = D[D.pid.isin(set(O["ids"]))].reset_index(drop=True)
    y, hard, w = D.S.values, D.hard.values, D.w.values
    pids = D.pid.values
    by = {p: np.where(O["ids"] == p)[0] for p in np.unique(O["ids"])}
    rng = np.random.default_rng(0)
    R, n = 30, 8
    print(f"{len(D)} rows / {D.pid.nunique()} problems, hard {hard.sum()} rows; train traces: "
          f"{'bench ' + str(len(B['ids'])) if B else 'LOPO official'}; n={n}, R={R}")
    # trace-level probe quality
    for c, T in enumerate(O["ck"]):
        alive = O["len"] > T
        if alive.sum() < 20:
            continue
        rows = []
        for kind in ("H", "M"):
            if O[kind] is None:
                continue
            for li, L in enumerate(O["ly"]):
                pr = probe_preds(O, B, c, li, kind)
                yy = np.log(O["len"][alive])
                r2 = 1 - np.mean((pr[alive] - yy) ** 2) / np.var(yy)
                # problem-level: mean over traces of (actual if done else max(pred, log T)) vs uncapped mean log len
                accs, aucs, aucs_cap = [], [], []
                for _ in range(R):
                    s_est, s_cap = {}, {}
                    for p, ix in by.items():
                        ix = rng.choice(ix, n, replace=False)
                        ln = O["len"][ix]
                        est = np.where(ln <= T, np.log(ln), np.maximum(pr[ix], np.log(T)))
                        s_est[p], s_cap[p] = est.mean(), np.log(np.minimum(ln, T)).mean()
                    se = np.array([s_est[p] for p in pids])
                    sc = np.array([s_cap[p] for p in pids])
                    aucs.append(auc(se[hard], ~y[hard]))
                    aucs_cap.append(auc(sc[hard], ~y[hard]))
                    pe = lopo_predict(se + 0.1 * w, D)
                    accs.append((np.mean(pe[hard] == y[hard]),
                                 np.mean(lopo_predict(sc + 0.1 * w, D)[hard] == y[hard])))
                a = np.mean(accs, 0)
                rows.append(f"T={T:5d} {kind} L={L:2d} alive {alive.sum():3d} R2 {r2:6.3f} | hard AUC est "
                            f"{np.mean(aucs):.3f} vs capped {np.mean(aucs_cap):.3f} | hard acc(+.1w) est {a[0]:.3f} "
                            f"vs capped {a[1]:.3f}")
        print("\n".join(rows), flush=True)
    full = np.array([np.log(O["len"][by[p]]).mean() for p in pids])
    print(f"uncapped mean log len over all traces: hard AUC {auc(full[hard], ~y[hard]):.3f}")


if __name__ == "__main__":
    main()
