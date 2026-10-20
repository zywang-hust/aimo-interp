"""Set decision.json to `readers` with z_thr = the median combined z over the public-validation cases.

Reader z values of the validation cases come from our own Codabench log (submission 1000206 logged every reader's
z per case; no problem text). The validation set is close to balanced, so the median splits it in half.
  python3 -I bench/scripts/val_median.py [--beta B] skywork-or1-7b__default [more readers ...]
With --beta, each case's score also gets B x the asked config's standardised weakness (1 - mean p_base on the
bench), as fragility.py does.
"""
import json
import re
import sys

import numpy as np

ROOT = "/user/wangzhiyao/projects/interp"
LOG = f"{ROOT}/submission/logs/1000206/logs.3.data_file"
ALL = ["qwen3.5-4b__default", "skywork-or1-7b__default", "olmo3-7b-think__default", "dsr1-qwen3-8b__default",
       "gpt-oss-120b__low", "gpt-oss-120b__medium", "gpt-oss-120b__high"]


KEYS = {"Qwen/Qwen3.5-4B": "qwen3.5-4b", "Skywork/Skywork-OR1-Math-7B": "skywork-or1-7b",
        "allenai/Olmo-3-7B-Think": "olmo3-7b-think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "dsr1-qwen3-8b",
        "openai/gpt-oss-120b": "gpt-oss-120b"}


def val_z():
    cases = []
    for line in open(LOG, errors="replace"):
        if line.startswith("[fragility]"):
            c = {k: float(v) for k, v in re.findall(r"(\S+__\w+)=\[([-0-9.e]+)\]", line)}
            mid, eff = re.match(r"\[fragility\] (\S+/\S+)/(\w+):", line).groups()
            c["cfg"] = f"{KEYS[mid]}__{eff if KEYS[mid] == 'gpt-oss-120b' else 'default'}"
            cases.append(c)
    return cases


def weakness():
    import pandas as pd
    w = {c: 1 - pd.read_csv(f"{ROOT}/bench/runs/probe/{c}.bench_ids.csv").p.mean() for c in ALL}
    v = np.array(list(w.values()))
    return {c: float((x - v.mean()) / v.std()) for c, x in w.items()}


def main():
    args = sys.argv[1:]
    beta = 0.0
    if args and args[0] == "--beta":
        beta, args = float(args[1]), args[2:]
    readers, cases, w = args, val_z(), weakness()
    z = np.array([np.mean([c[r] for r in readers]) + beta * w[c["cfg"]] for c in cases])
    thr = float(np.median(z))
    d = {"readers": readers, "z_thr": thr, "beta": beta, "weakness": w,
         "log_readers": [r for r in ALL if r not in readers],
         "note": f"z_thr = median combined z over the {len(cases)} public-validation cases (submission 1000206 log)"}
    json.dump(d, open(f"{ROOT}/submission/fragility/decision.json", "w"), indent=1)
    print(f"{len(cases)} cases; z_thr {thr:.3f}; {int((z > thr).sum())} flagged spurious")


if __name__ == "__main__":
    main()
