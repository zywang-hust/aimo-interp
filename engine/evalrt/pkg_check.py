"""Run the submission package the way Codabench does: one fresh process per case, one problem per call.
Each case is a subprocess (python -I) that imports submission/fragility/solution.py from a copy of the package dir and
calls are_robust(model_id, effort, [problem]); its stderr (the [fragility] line) goes to the log, with the problem
text never printed by this script. Prints per-case wall time and the decision.
  GPU=<id> engine/evalrt/run_eval.sh engine/evalrt/pkg_check.py [--config <decision.json>] <case spec> ...
case spec: <pid>:<model_id>:<effort>, e.g. 388d7602:openai/gpt-oss-120b:high
--config replaces decision.json in the copy (e.g. submission/configs/probe1536_v2.json); default: the package's own.
State (start time, cached rollouts) goes to a per-run dir, so parallel or recent runs on this host (/tmp, /dev/shm)
do not hand each other cached rollouts; FRAGILITY_STATE_DIR overrides it.
PKG_COLD_TRITON=1: every case gets an empty TRITON_CACHE_DIR (in the work dir), as on a fresh eval container.
"""
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = "/user/wangzhiyao/projects/interp"
CASE = """
import json, os, sys
if os.environ.get("EVAL_EXTRA_SITE"):  # kernels overlay of the eval-image replica (MXFP4 kernels)
    sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])
sys.path.insert(0, sys.argv[1])
from solution import are_robust
a = json.load(open(sys.argv[2]))
print("RESULT", json.dumps(are_robust(a["model_id"], a["effort"], [a["problem"]])), flush=True)
"""


def main():
    probs = {r["id"]: r["problem"] for r in map(json.loads, open(f"{ROOT}/bench/train36_input.jsonl"))}
    work = f"/user/wangzhiyao/cache/tmp/pkg_check_{os.getpid()}"
    pkg = f"{work}/ingested_program"
    shutil.copytree(f"{ROOT}/submission/fragility", pkg, ignore=shutil.ignore_patterns("__pycache__", ".fragility*"))
    args = sys.argv[1:]
    if args[:1] == ["--config"]:
        shutil.copy(args[1], f"{pkg}/decision.json")
        print("config", args[1], json.load(open(args[1]))["rollout"].get("engine", "v1"), flush=True)
        args = args[2:]
    env = {**os.environ, "FRAGILITY_MODEL_ROOT": "/user/wangzhiyao/models/aimo"}
    env.setdefault("FRAGILITY_STATE_DIR", f"{work}/state")
    env["HOME"] = f"{work}/home"  # never the real /root
    os.makedirs(env["HOME"], exist_ok=True)
    if env.get("EVAL_EXTRA_SITE"):
        env["EVAL_EXTRA_SITE"] = os.path.abspath(env["EVAL_EXTRA_SITE"])
    t_job = time.time()
    for i, spec in enumerate(args):
        pid, model_id, effort = spec.split(":")
        arg = f"{work}/case{i}.json"
        json.dump({"model_id": model_id, "effort": effort, "problem": probs[pid]}, open(arg, "w"))
        if os.environ.get("PKG_COLD_TRITON"):
            env["TRITON_CACHE_DIR"] = f"{work}/triton{i}"
        t0 = time.time()
        p = subprocess.run([sys.executable, "-I", "-c", CASE, pkg, arg], env=env, capture_output=True, text=True,
                           cwd=work)
        res = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        frag = [l for l in p.stderr.splitlines() if l.startswith("[fragility]")]
        print(f"case {i} {pid} {model_id}/{effort}: rc {p.returncode} {time.time() - t0:.1f}s "
              f"(job {time.time() - t_job:.0f}s) {res}", flush=True)
        for l in frag:
            print("   ", l[:3000], flush=True)
        if p.returncode:
            print(p.stderr[-3000:], flush=True)
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
