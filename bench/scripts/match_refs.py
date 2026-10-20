"""Match train-main-v2 problems to known contest problems (MathArena sets + AIMO2/3 reference).

exact=True means identical after whitespace normalisation, so the reference answer applies.
Near matches (sim > 0.8 but not exact) are number-modified twins: their answer must be recomputed.
"""
import difflib
import glob
import re

import pandas as pd

ROOT = "/user/wangzhiyao/projects/interp"


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def tok(s):
    return set(re.findall(r"[A-Za-z]+|\d+", s))


def load_refs():
    refs = []
    for f in glob.glob(f"{ROOT}/data/ext_answers/*.parquet"):
        x = pd.read_parquet(f)
        src = f.split("__")[1]
        refs += [(src, str(r.problem), str(r.answer)) for r in x.itertuples()]
    for name in ("aimo3", "aimo2"):
        x = pd.read_csv(f"{ROOT}/data/kaggle/{name}/reference.csv")
        refs += [(name, str(r.problem), str(r.answer)) for r in x.itertuples()]
    return refs


def main():
    refs = load_refs()
    P = pd.read_json(f"{ROOT}/bench/train36_problems.jsonl", lines=True)
    out = []
    for r in P.itertuples():
        T = tok(r.problem)
        best = max(refs, key=lambda q: len(T & tok(q[1])) / len(T | tok(q[1])))
        sim = difflib.SequenceMatcher(None, norm(r.problem), norm(best[1])).ratio()
        out.append(dict(pid=r.pid, src=best[0], sim=round(sim, 3), ref_answer=best[2],
                        exact=norm(r.problem) == norm(best[1])))
    o = pd.DataFrame(out).sort_values("sim", ascending=False)
    o.to_json(f"{ROOT}/bench/train36_refmatch.jsonl", orient="records", lines=True)
    print(o.to_string())
    print(f"{o.exact.sum()} exact, {((o.sim > 0.8) & ~o.exact).sum()} near-twins, {(o.sim <= 0.8).sum()} unmatched")


if __name__ == "__main__":
    main()
