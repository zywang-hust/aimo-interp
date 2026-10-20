"""Per-problem decision threshold for the submission, fitted on the official labelled rows.

The real ingestion calls are_robust once per problem, in a fresh process (submission 1000171), so the decision
must be an absolute threshold on one problem's score, not a within-batch rank. Score = a reader's length-probe
prediction for the problem, computed from bench/feats/train36 with the exported artifact (the same numbers the
submission computes). Readers are combined as the mean of their scores standardised by each reader's bench
reference distribution. The threshold maximises balanced accuracy (the validation set is close to balanced);
LOPO = leave one problem out, all its rows together.
  python3 -I bench/scripts/calibrate.py [reader-cfg ...]     (default: every artifact, each alone)
  python3 -I bench/scripts/calibrate.py --write qwen3.5-4b__default   -> submission/fragility/decision.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402
from common import auc as common_auc  # noqa: E402

ALL = ["qwen3.5-4b__default", "skywork-or1-7b__default", "olmo3-7b-think__default", "dsr1-qwen3-8b__default",
       "gpt-oss-120b__low", "gpt-oss-120b__medium", "gpt-oss-120b__high"]


def reader_scores(cfg):
    a = np.load(f"{ROOT}/submission/fragility/artifacts/{cfg}.npz")
    f = np.load(f"{ROOT}/bench/feats/train36/{cfg}.npz", allow_pickle=True)
    h = f["last"][:, int(a["layer"])].astype(np.float64)
    s = ((h - a["mu"]) / a["sd"]) @ a["w"] + a["ym"]
    ref = a["ref"]
    return dict(zip(f["ids"], (s - ref.mean()) / ref.std()))


def auc(s, robust):
    """P(score of a spurious row > score of a robust row); `robust` is the is_robust mask."""
    return common_auc(s, ~np.asarray(robust, bool))


def bal(pred_robust, y):
    return 0.5 * ((pred_robust & y).sum() / y.sum() + (~pred_robust & ~y).sum() / (~y).sum())


def best_thr(s, y):
    cands = np.unique(s)
    cands = np.r_[cands[0] - 1, (cands[:-1] + cands[1:]) / 2, cands[-1] + 1]
    b = [bal(s <= t, y) for t in cands]
    i = int(np.argmax(b))
    return cands[i], b[i]


def evaluate(cfgs, rows):
    sc = [reader_scores(c) for c in cfgs]
    s = np.array([np.mean([d[r["pid"]] for d in sc]) for r in rows])
    y = np.array([r["is_robust"] for r in rows])
    pid = np.array([r["pid"] for r in rows])
    margin = np.array([min(abs(r["max_drop"] - 0.10), abs(r["max_drop"] - 0.25)) for r in rows])
    pred = np.zeros(len(rows), bool)
    for p in np.unique(pid):
        m = pid == p
        t, _ = best_thr(s[~m], y[~m])
        pred[m] = s[m] <= t
    t, b = best_thr(s, y)
    far = margin >= 0.10
    return dict(readers="+".join(c.split("__")[0] if c.endswith("default") else c for c in cfgs),
                auc=auc(s, y), auc_far=auc(s[far], y[far]), lopo_acc=(pred == y).mean(), lopo_bal=bal(pred, y),
                lopo_bal_far=bal(pred[far], y[far]), thr=t, insample_bal=b)


def main():
    rows = [r for r in map(json.loads, open(f"{ROOT}/bench/train_main_v2_rows.jsonl")) if r["is_robust"] is not None]
    args = [a for a in sys.argv[1:] if a != "--write"]
    sets = [a.split("+") for a in args] or [[c] for c in ALL]
    print(f"{len(rows)} rows, {sum(not r['is_robust'] for r in rows)} spurious")
    for cfgs in sets:
        r = evaluate(cfgs, rows)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    if "--write" in sys.argv:
        out = f"{ROOT}/submission/fragility/decision.json"
        json.dump({"readers": sets[0], "z_thr": float(r["thr"]), "lopo_bal": round(float(r["lopo_bal"]), 3),
                   "rows": len(rows)}, open(out, "w"), indent=1)
        print("wrote", out)


if __name__ == "__main__":
    main()
