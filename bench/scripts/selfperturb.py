"""Self-perturbation: does gpt-oss-medium think longer / disagree more on gpt-oss-made variants of a problem, beyond
how long it thinks on the problem itself?  Problem-level (labels are a property of the problem), official rows.
Variants: bench/runs/step2 perturb_*.jsonl (rule typos; gpt-oss rename/rephrase/distract/domain/expert_*),
medium samples from step2/out + step2b/out (8 per variant, uncapped 32k). Base: train36all medium (16 samples).
  python3 -I bench/scripts/selfperturb.py            # full-sample signal per family
  python3 -I bench/scripts/selfperturb.py --eval     # at eval-time sample sizes and caps (what the package can run)
  python3 -I bench/scripts/selfperturb.py --fit G [T]  # threshold of the rule base - G*d + c*w (submission 1000708)
Dropped on 2026-10-19 (NOTES §5): d is redundant with the base length; the package no longer runs it.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import auc  # noqa: E402
from evalbench import load, lopo_logit, lopo_predict  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"


def main():
    D, runs = load()
    P = D.groupby("pid").agg(S=("S", "mean"), hard=("hard", "max"), w=("w", "mean")).reset_index()
    P["S"] = P.S > 0.5
    O = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/bench/runs/{d}/out/gpt-oss-120b__medium.jsonl")])
                   for d in ("step2", "step2b")])
    O = O.drop_duplicates(["id", "k"])
    O["pid"] = O.id.str.split("|").str[0]
    O["fam"] = O.id.str.split("|").str[1]
    O["ll"] = np.log(O.completion_tokens)
    keys = {}
    for f in ("perturb_train36", "perturb_train36_expert", "perturb_train36_modify"):
        for r in map(json.loads, open(f"{ROOT}/bench/runs/step2/{f}.jsonl")):
            keys[f"{r['id']}|{r['family']}|{r['j']}"] = str(r.get("answer"))
    O["key"] = O.id.map(keys)
    O["ok"] = (O.answer.astype(str) == O.key) & O.key.notna() & (O.key != "None")
    base = {p: np.log(v).mean() for p, v in runs["medium"].items()}
    P["base"] = P.pid.map(base)
    hard, y = P.hard.values, P.S.values
    print(f"{len(P)} problems, hard {hard.sum()} ({y[hard].sum()} S); base mean log len: AUC all {auc(P.base.values, ~y):.3f} "
          f"hard {auc(P.base.values[hard], ~y[hard]):.3f}")
    print(f"{'family':22s} {'n':>3s} {'d AUC all':>10s} {'d AUC hard':>11s} {'resid hard':>11s} {'acc(var) hard':>14s}")
    feats = {}
    for fam, g in O.groupby("fam"):
        m = g.groupby("pid").ll.mean()
        a = g.groupby("pid").ok.mean()
        sub = P[P.pid.isin(m.index)]
        b, yy, hh = sub.base.values, sub.S.values, sub.hard.values
        d = m[sub.pid].values - b
        A = np.c_[np.ones_like(b), b]
        r = d - A @ np.linalg.lstsq(A, d, rcond=None)[0]
        feats[fam] = dict(zip(sub.pid, d))
        acc = a[sub.pid].values
        print(f"{fam:22s} {len(sub):3d} {auc(d, ~yy):10.3f} {auc(d[hh], ~yy[hh]):11.3f} {auc(r[hh], ~yy[hh]):11.3f} "
              f"{auc(-acc[hh], ~yy[hh]):14.3f}")
    # combined: mean over the LLM families of d, residualized
    fams = ["rename", "rephrase", "distract", "domain", "expert_modify", "typos"]
    dm = np.array([np.nanmean([feats[f].get(p, np.nan) for f in fams]) for p in P.pid])
    ok = ~np.isnan(dm)
    print(f"mean d over {fams}: problems {ok.sum()}, AUC all {auc(dm[ok], ~y[ok]):.3f} hard {auc(dm[ok & hard], ~y[ok & hard]):.3f}")
    Dk = P[ok].reset_index(drop=True)
    for name, X in [("base", np.c_[Dk.base]), ("base+d", np.c_[Dk.base, dm[ok]])]:
        Dk2 = Dk.assign(pid=Dk.pid)
        pr = lopo_logit(X, Dk2)
        h = Dk.hard.values
        print(f"LOPO logit {name:8s}: acc all {np.mean((pr > .5) == Dk.S.values):.3f} hard "
              f"{np.mean((pr[h] > .5) == Dk.S.values[h]):.3f} AUC hard {auc(pr[h], ~Dk.S.values[h]):.3f}")


def eval_time(fams=("domain", "rename"), n_base=8, n_var=4, k_var=1, Ts=(2048, 3072, 4096, 6144, 8192, 32768),
              reps=200, seed=0):
    """Eval-box version, row level like the package: base = mean log min(len, T) of n_base medium samples; per family
    k_var random variants with n_var samples each, d = mean over families of its capped mean log length - base.
    LOPO logistic on [base, w] vs [base, d, w]; mean accuracy (all / hard rows) over reps of the sampling noise."""
    D, runs = load()
    O = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/bench/runs/{d}/out/gpt-oss-120b__medium.jsonl")])
                   for d in ("step2", "step2b")]).drop_duplicates(["id", "k"])
    O["pid"], O["fam"] = O.id.str.split("|").str[0], O.id.str.split("|").str[1]
    var = {(p, f): [g.completion_tokens.values for _, g in gf.groupby("id") if len(g) >= n_var]
           for (p, f), gf in O[O.fam.isin(fams)].groupby(["pid", "fam"])}
    D = D[[all(var.get((p, f)) for f in fams) for p in D.pid]].reset_index(drop=True)
    h, y, w = D.hard.values, D.S.values, D.w.values
    print(f"{D.pid.nunique()} problems / {len(D)} rows (hard {h.sum()}, {y[h].sum()} S), families {fams}, "
          f"base n={n_base}, {k_var} variant(s) per family x n={n_var}, {reps} reps")
    rng = np.random.default_rng(seed)
    for T in Ts:
        acc = {"base+w": [], "base+d+w": []}
        auc_d = []
        for _ in range(reps):
            b = {p: np.log(np.minimum(rng.choice(v, n_base, replace=False), T)).mean() for p, v in runs["medium"].items()}
            dd = {}
            for p in D.pid.unique():
                vs = [v[j] for v in (var[(p, f)] for f in fams) for j in rng.choice(len(v), min(k_var, len(v)), replace=False)]
                dd[p] = np.mean([np.log(np.minimum(rng.choice(s, n_var, replace=False), T)).mean() for s in vs]) - b[p]
            base, d = D.pid.map(b).values, D.pid.map(dd).values
            auc_d.append(auc(d[h], ~y[h]))
            for name, X in (("base+w", np.c_[base, w]), ("base+d+w", np.c_[base, d, w])):
                pr = lopo_logit(X, D) > 0.5
                acc[name].append(((pr == y).mean(), (pr[h] == y[h]).mean()))
        line = "  ".join(f"{k} all {np.mean(v, 0)[0]:.3f} hard {np.mean(v, 0)[1]:.3f}" for k, v in acc.items())
        print(f"T={T:5d}: d AUC(Robust high) hard {np.mean(auc_d):.3f}±{np.std(auc_d):.3f} | {line}")


def fit(gamma, fams=("domain", "rename"), n_base=8, n_var=4, T=4096, c=0.1, reps=400, seed=0):
    """Threshold of the package rule score = base - gamma*d + c*w (S if above), maximising balanced accuracy over all
    official rows pooled over reps of the eval-time sampling noise; also its LOPO accuracy per rep."""
    D, runs = load()
    O = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/bench/runs/{d}/out/gpt-oss-120b__medium.jsonl")])
                   for d in ("step2", "step2b")]).drop_duplicates(["id", "k"])
    O["pid"], O["fam"] = O.id.str.split("|").str[0], O.id.str.split("|").str[1]
    var = {(p, f): [g.completion_tokens.values for _, g in gf.groupby("id") if len(g) >= n_var]
           for (p, f), gf in O[O.fam.isin(fams)].groupby(["pid", "fam"])}
    y, w, h = D.S.values, D.w.values, D.hard.values
    rng = np.random.default_rng(seed)
    S, acc = [], []
    for _ in range(reps):
        b = {p: np.log(np.minimum(rng.choice(v, n_base, replace=False), T)).mean() for p, v in runs["medium"].items()}
        dd = {p: np.mean([np.log(np.minimum(rng.choice(v[rng.integers(len(v))], n_var, replace=False), T)).mean()
                          for v in (var[(p, f)] for f in fams)]) - b[p] for p in D.pid.unique()}
        s = D.pid.map(b).values - gamma * D.pid.map(dd).values + c * w
        S.append(s)
        pr = lopo_predict(s, D)
        acc.append(((pr == y).mean(), (pr[h] == y[h]).mean()))
    S = np.concatenate(S)
    Y = np.tile(y, reps)
    cand = np.quantile(S, np.linspace(0.02, 0.98, 400))
    bal = [0.5 * ((S[Y] > t).mean() + (S[~Y] <= t).mean()) for t in cand]
    t = float(cand[int(np.argmax(bal))])
    H = np.tile(h, reps)
    print(f"gamma {gamma} T {T} c {c}: thr {t:.4f} (pooled balanced acc {max(bal):.3f}, hard-row acc at thr "
          f"{((S[H] > t) == Y[H]).mean():.3f}); LOPO acc all {np.mean(acc, 0)[0]:.3f} hard {np.mean(acc, 0)[1]:.3f}")


if __name__ == "__main__":
    if "--fit" in sys.argv:
        a = sys.argv[sys.argv.index("--fit") + 1:]
        fit(float(a[0]), T=int(a[1]) if len(a) > 1 else 4096)
    else:
        eval_time() if "--eval" in sys.argv else main()
