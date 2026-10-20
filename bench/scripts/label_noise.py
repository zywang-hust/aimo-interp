"""Finite-sample model of the max_drop label.

base: k_b ~ Bin(n_b, p); family f: k_f ~ Bin(n_f, p - delta_f).
max_drop = k_b/n_b - min_f k_f/n_f.  R iff <=0.10, S iff >=0.25, else excluded.
Exact computation via binomial pmfs (families iid given p).
"""
import numpy as np
from scipy.stats import binom

LO, HI = 0.10, 0.25
EPS = 1e-9


def probs(p, n_b, n_f, F, delta=0.0, worst_delta=None):
    """P(R), P(S), P(band). delta applies to all families; worst_delta (if given) to one family,
    the remaining F-1 families get `delta`."""
    kb = np.arange(n_b + 1)
    pb = binom.pmf(kb, n_b, p)
    kf = np.arange(n_f + 1)
    q = np.clip(p - delta, 0, 1)
    # CDF of min over families: P(min >= x) = prod P(k_f >= x)
    sf = binom.sf(kf - 1, n_f, q)  # P(k >= kf)
    if worst_delta is None:
        surv = sf ** F
    else:
        q2 = np.clip(p - worst_delta, 0, 1)
        sf2 = binom.sf(kf - 1, n_f, q2)
        surv = sf ** (F - 1) * sf2
    pmin = surv - np.append(surv[1:], 0.0)  # P(min == kf)
    d = kb[:, None] / n_b - kf[None, :] / n_f
    w = pb[:, None] * pmin[None, :]
    R = w[d <= LO + EPS].sum()
    S = w[d >= HI - EPS].sum()
    return R, S, 1 - R - S


def table(designs, ps, **kw):
    out = {}
    for name, (nb, nf, F) in designs.items():
        rows = []
        for p in ps:
            R, S, B = probs(p, nb, nf, F, **kw)
            lab = R + S
            rows.append((p, R, S, B, S / lab if lab > 0 else np.nan,
                         max(R, S) / lab if lab > 0 else np.nan))
        out[name] = np.array(rows)
    return out


if __name__ == "__main__":
    designs = {
        "nb10_nf10_F5": (10, 10, 5),
        "nb10_nf10_F7": (10, 10, 7),
        "nb32_nf32_F6": (32, 32, 6),
        "nb100_nf100_F5": (100, 100, 5),
        "nb100_nf100_F7": (100, 100, 7),
    }
    ps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    print("=== NULL (delta=0): columns p, P(R), P(S), P(band), P(S|labeled), ceiling-acc|p")
    for name, arr in table(designs, ps).items():
        print("--", name)
        for r in arr:
            print("  p=%.2f R=%.3f S=%.3f band=%.3f S|lab=%.3f acc|p=%.3f" % tuple(r))
    print("\n=== E[max_drop] under null")
    rng = np.random.default_rng(0)
    for name, (nb, nf, F) in designs.items():
        s = []
        for p in [0.3, 0.5, 0.7, 0.9]:
            kb = rng.binomial(nb, p, 200000) / nb
            kf = rng.binomial(nf, p, (200000, F)).min(1) / nf
            s.append("p=%.1f E=%.3f sd=%.3f" % (p, (kb - kf).mean(), (kb - kf).std()))
        print(name, "; ".join(s))

    print("\n=== Power: one family has true drop delta*, others 0. P(R),P(S),P(band)")
    for name, (nb, nf, F) in designs.items():
        print("--", name)
        for p in [0.5, 0.8, 0.95]:
            for dw in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
                if dw > p:
                    continue
                R, S, B = probs(p, nb, nf, F, worst_delta=dw)
                print("  p=%.2f d*=%.2f R=%.3f S=%.3f band=%.3f" % (p, dw, R, S, B))
