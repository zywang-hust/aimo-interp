"""Join our base-accuracy sampling of the official problems (bench/runs/step1) to the official rows.

Writes bench/runs/step1_joined.csv (official row + n, p_base, trunc, mean_tok of the asked config) and prints the
per-config summary and how p_base relates to the labels.
  python3 -I bench/scripts/join_step1.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_KEYS, ROOT, auc, load_samples  # noqa: E402


def main():
    S = load_samples(f"{ROOT}/bench/runs/step1/out/*.jsonl")
    ref = pd.read_json(f"{ROOT}/bench/train36_answers.jsonl", lines=True, dtype=False).set_index("id").answer.astype(str)
    S["correct"] = [a is not None and str(a) == ref.get(i) for a, i in zip(S.answer, S.id)]
    S["model_id"] = S.model.map(MODEL_KEYS).fillna(S.model)
    per = S.groupby(["model_id", "effort", "id"]).agg(
        n=("correct", "size"), p_base=("correct", "mean"), trunc=("finish", lambda f: (f == "length").mean()),
        mean_tok=("completion_tokens", "mean")).reset_index()
    print(per.groupby(["model_id", "effort"])[["n", "p_base", "trunc", "mean_tok"]].mean().round(3).to_string())

    T = pd.read_json(f"{ROOT}/bench/train_main_v2_rows.jsonl", lines=True).rename(columns={"pid": "id"})
    J = T.merge(per, left_on=["model_id", "reasoning_effort", "id"], right_on=["model_id", "effort", "id"], how="left")
    J.to_csv(f"{ROOT}/bench/runs/step1_joined.csv", index=False)
    L = J[J.is_robust.notna() & J.p_base.notna()]
    sp = ~L.is_robust.astype(bool)
    print(f"{J.p_base.notna().sum()}/{len(J)} official rows have p_base; AUC(Spurious | 1 - p_base) "
          f"{auc(1 - L.p_base, sp):.3f}")


if __name__ == "__main__":
    main()
