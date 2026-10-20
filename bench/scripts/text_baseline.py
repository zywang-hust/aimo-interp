"""Ablation baseline without any model internals: predict a config's log mean completion length from the problem
text alone, trained on the same bench pool problems as train_probe.py, then transferred to the official problems
and scored as a Spurious detector (same protocol and outputs as the white-box probe).

Features: TF-IDF of word 1-2grams + char 3-5grams, plus hand features (length, #numbers, #math segments,
max number magnitude). Ridge, alpha by grouped CV.
  python3 -I bench/scripts/text_baseline.py --model-key qwen3.5-4b [--effort default] [--twins] [--min-n 2]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = "/user/wangzhiyao/projects/interp"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_KEYS, auc  # noqa: E402
from train_probe import spearman, targets  # noqa: E402


def hand(texts):
    rows = []
    for t in texts:
        nums = re.findall(r"\d+", t)
        rows.append([np.log1p(len(t)), len(nums), len(re.findall(r"\$", t)) / 2,
                     np.log1p(max([int(n[:15]) for n in nums], default=0))])
    return np.array(rows, float)


def featurize(train_texts):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train_texts)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(train_texts)
    H = hand(train_texts)
    mu, sd = H.mean(0), H.std(0) + 1e-6
    return lambda T: hstack([w.transform(T), c.transform(T), csr_matrix((hand(T) - mu) / sd)]).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--effort", default="default")
    ap.add_argument("--twins", action="store_true")
    ap.add_argument("--min-n", type=int, default=4)
    args = ap.parse_args()
    cfg = f"{args.model_key}__{args.effort}"
    ans = {r["id"]: str(r["answer"]) for r in map(json.loads, open(f"{ROOT}/bench/pool/answers.jsonl"))}
    parent = {r["id"]: r.get("parent", r["id"]) for r in map(json.loads, open(f"{ROOT}/bench/pool/answers.jsonl"))}
    text = {}
    for f in ["bench/pool/pool_input.jsonl", "bench/pool/twins_verified.jsonl"]:
        text.update({r["id"]: r["problem"] for r in map(json.loads, open(f"{ROOT}/{f}"))})
    Ys = [targets(f"{ROOT}/bench/runs/pool/out/{cfg}*.jsonl", ans)]
    if args.twins:
        Ys.append(targets(f"{ROOT}/bench/runs/twins/out/{cfg}*.jsonl", ans))
    Y = pd.concat(Ys)
    Y = Y[Y.n >= args.min_n]
    ids = list(Y.index)
    y = Y.loglen.values
    groups = np.array([parent[i] for i in ids])
    texts = [text[i] for i in ids]

    ug = np.random.default_rng(0).permutation(np.unique(groups))
    best = None
    for alpha in [0.3, 1, 3, 10, 30]:
        pred = np.zeros(len(y))
        for f in range(5):
            te = np.isin(groups, ug[f::5])
            fz = featurize([t for t, m in zip(texts, ~te) if m])
            pred[te] = Ridge(alpha=alpha).fit(fz([t for t, m in zip(texts, ~te) if m]), y[~te]).predict(
                fz([t for t, m in zip(texts, te) if m]))
        s = spearman(pred, y)
        if best is None or s > best[0]:
            best = (s, alpha, pred.copy())
    print(f"{cfg}: {len(y)} problems; text-only CV spearman {best[0]:.3f} (alpha {best[1]})")

    fz = featurize(texts)
    model = Ridge(alpha=best[1]).fit(fz(texts), y)
    T = pd.read_json(f"{ROOT}/bench/train_main_v2_rows.jsonl", lines=True)
    T = T[(T.model_id == MODEL_KEYS[args.model_key]) & (T.reasoning_effort == args.effort)]
    T = T.assign(pred=model.predict(fz(list(T.problem))))
    lab = T[T.is_robust.notna()]
    print(f"  transfer AUC(Spurious | text-only prediction) = {auc(lab.pred, lab.is_robust == 0):.3f} on {len(lab)} rows")
    os.makedirs(f"{ROOT}/bench/runs/probe", exist_ok=True)
    T.to_csv(f"{ROOT}/bench/runs/probe/{cfg}.loglen.text.csv", index=False)
    np.save(f"{ROOT}/bench/runs/probe/{cfg}.loglen.text.benchcv.npy", best[2])
    pd.DataFrame({"id": ids, "p": Y.p.values}).to_csv(f"{ROOT}/bench/runs/probe/{cfg}.text.bench_ids.csv", index=False)


if __name__ == "__main__":
    main()
