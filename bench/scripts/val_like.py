"""Which signals still separate Robust from Spurious on hard ("validation-like") official rows?

The public validation problems are hard: 20 of 21 have qwen length z > -0.42 (submission 1000189/1000206 logs),
and the qwen length ranking scored at chance there (1000206). The official rows with qwen z > -0.42 are the
closest labelled proxy (about 58% Spurious, like the balanced validation set). AUC = P(score of a Spurious row >
score of a Robust row); < 0.5 means the signal is reversed.
  python3 -I bench/scripts/val_like.py [z_cut]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_KEYS  # noqa: E402
from calibrate import ALL, auc, reader_scores  # noqa: E402

ROOT = "/user/wangzhiyao/projects/interp"
KEY = {v: k for k, v in MODEL_KEYS.items()}


def table():
    rows = [r for r in map(json.loads, open(f"{ROOT}/bench/train_main_v2_rows.jsonl")) if r["is_robust"] is not None]
    D = pd.DataFrame(rows)
    D["cfg"] = [f"{KEY[m]}__{e}" for m, e in zip(D.model_id, D.reasoning_effort)]
    sc = {c: reader_scores(c) for c in ALL}
    for c in ALL:
        D[f"z:{c.replace('__default', '')}"] = D.pid.map(sc[c])
    D["own_len"] = [sc[c][p] for c, p in zip(D.cfg, D.pid)]
    P = {}
    for f in glob.glob(f"{ROOT}/bench/runs/probe/*.p.last.csv"):
        c = os.path.basename(f).split(".p.last")[0]
        d = pd.read_csv(f)
        P.update({(c, pid): v for pid, v in zip(d.pid, d.pred)})
    D["own_1-p_probe"] = [1 - P.get((c, p), np.nan) for c, p in zip(D.cfg, D.pid)]
    s1 = pd.read_csv(f"{ROOT}/bench/runs/step1_joined.csv")[["problem", "model_id", "reasoning_effort", "p_base", "mean_tok"]]
    D = D.merge(s1, on=["problem", "model_id", "reasoning_effort"], how="left")
    D["measured_1-p"] = 1 - D.p_base
    D["measured_rel_len"] = D.groupby("cfg").mean_tok.transform(lambda x: np.log(x / x.median()))
    D["measured_len"] = np.log(D.mean_tok)
    D["chars"] = D.problem.str.len()
    D["digits"] = D.problem.str.count(r"\d")
    return D


def main():
    cut = float(sys.argv[1]) if len(sys.argv) > 1 else -0.42
    D = table()
    y = D.is_robust.values.astype(bool)
    hard = D["z:qwen3.5-4b"].values > cut
    print(f"{len(D)} rows; hard (qwen z > {cut}): {hard.sum()} rows, {D[hard].pid.nunique()} problems, "
          f"{(~y[hard]).mean():.2f} Spurious")
    cols = [c for c in D.columns if c.startswith(("z:", "own_", "measured_")) or c in ("chars", "digits")]
    for c in cols:
        v = D[c].values.astype(float)
        ok = ~np.isnan(v)
        print(f"  {c:26s} all {auc(v[ok], y[ok]):.3f}   hard {auc(v[ok & hard], y[ok & hard]):.3f}  (n={int((ok & hard).sum())})")


if __name__ == "__main__":
    main()
