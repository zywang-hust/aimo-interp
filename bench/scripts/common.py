"""Shared helpers for the bench scripts."""
import glob

import numpy as np
import pandas as pd

ROOT = "/user/wangzhiyao/projects/interp"

MODEL_KEYS = {  # served model name (engine/models.yaml keys) -> official model_id
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "skywork-or1-7b": "Skywork/Skywork-OR1-Math-7B",
    "olmo3-7b-think": "allenai/Olmo-3-7B-Think",
    "dsr1-qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}


def auc(score, y):
    """P(score of a positive > score of a negative), ties count 1/2."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def load_samples(pattern: str) -> pd.DataFrame:
    """sample.py outputs matching a glob, without pure-error rows; rows sampled before the extractor fix get their
    answer re-parsed from the stored raw ANSWER-line capture."""
    from sample import extract_answer
    S = pd.concat([pd.read_json(f, lines=True, dtype=False) for f in glob.glob(pattern)], ignore_index=True)
    S = S[S.answer.notna() | S.finish.notna()].copy()
    miss = S.answer.isna() & S.raw_answer.notna()
    S.loc[miss, "answer"] = S.loc[miss, "raw_answer"].map(lambda r: extract_answer("ANSWER: " + str(r))[1])
    return S
