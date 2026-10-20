"""Build the candidate problem pool (integer-answer competition problems) from MathArena + AIMO references.

Writes data/pool/pool_raw.jsonl: {id, source, problem, answer}. Integer answers only (the official task scores
integer answers). Problems already in train-main-v2 are tagged in_train=True (kept for factory calibration).
"""
import hashlib
import json
import os
import re
import urllib.request

import pandas as pd

ROOT = "/user/wangzhiyao/projects/interp"
OUT = f"{ROOT}/data/pool"
SETS = ["aime_2023_I", "aime_2023_II", "aime_2024_I", "aime_2024_II", "aime_2025", "aime_2026",
        "hmmt_feb_2023", "hmmt_feb_2024", "hmmt_feb_2025", "hmmt_feb_2026", "hmmt_nov_2025",
        "smt_2025", "brumo_2025", "cmimc_2025", "apex_2025", "apex-shortlist", "final_answer_comps"]


def hf_get(path):
    tok = open("/user/wangzhiyao/cache/huggingface/token").read().strip()
    req = urllib.request.Request(f"https://huggingface.co/{path}", headers={"Authorization": f"Bearer {tok}"})
    return urllib.request.urlopen(req, timeout=60).read()


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    rows = []
    for name in SETS:
        try:
            tree = json.loads(hf_get(f"api/datasets/MathArena/{name}/tree/main?recursive=true"))
        except Exception as e:
            print("skip", name, e)
            continue
        for f in [x["path"] for x in tree if x["path"].endswith(".parquet")]:
            local = f"{OUT}/raw/{name}__{os.path.basename(f)}"
            if not os.path.exists(local):
                open(local, "wb").write(hf_get(f"datasets/MathArena/{name}/resolve/main/{f}"))
            x = pd.read_parquet(local)
            if "problem" not in x or "answer" not in x:
                print("no problem/answer cols", name, list(x.columns))
                continue
            src_col = "source" if "source" in x.columns else None
            for r in x.itertuples():
                src = f"{name}:{getattr(r, 'source')}" if src_col else name
                rows.append(dict(source=src, problem=str(r.problem), answer=str(r.answer).strip()))
    for name in ("aimo2", "aimo3"):
        x = pd.read_csv(f"{ROOT}/data/kaggle/{name}/reference.csv")
        rows += [dict(source=name, problem=str(r.problem), answer=str(r.answer)) for r in x.itertuples()]
    df = pd.DataFrame(rows)
    df = df[df.answer.str.fullmatch(r"-?\d+")]
    df["key"] = df.problem.map(norm)
    df = df.drop_duplicates("key")
    df["id"] = df.key.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:8])
    train = {norm(json.loads(l)["problem"]) for l in open(f"{ROOT}/bench/train36_problems.jsonl")}
    df["in_train"] = df.key.isin(train)
    df = df.drop(columns="key")
    df.to_json(f"{OUT}/pool_raw.jsonl", orient="records", lines=True, force_ascii=False)
    print(len(df), "integer-answer problems;", df.in_train.sum(), "in train-main-v2")
    print(df.source.str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
