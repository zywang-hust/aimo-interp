#!/usr/bin/env bash
# Sample the 4 small models on the accepted no-solution variants (expected answer NaN), n=4 per variant,
# against the servers on idx3 (qwen 31201, skywork 31202, dsr1 31203) and idx4 (olmo 31204).
# Runs in rounds while generation is still going: each round samples every variant accepted so far
# (sample.py resumes, so done pairs are skipped); stops after a round started once generation had finished.
#   bench/runs/nosol/run_sample.sh <gen-pid>
set -uo pipefail
GEN=$1
R=/user/wangzhiyao/projects/interp
S=$R/bench/runs/nosol
export NO_PROXY='*' no_proxy='*'
mkdir -p $S/logs
run() {  # key url concurrency maxtok
  python3 -I $R/bench/scripts/sample.py --input $S/inputs/variants.jsonl --out $S/out/$1__default.jsonl \
    --model $1 --base-url $2 --n 4 --max-tokens $4 --concurrency $3 >> $S/logs/$1.log 2>&1
}
while :; do
  gen_done=0; kill -0 "$GEN" 2>/dev/null || gen_done=1
  python3 -I - <<'EOF'
import json
S = "/user/wangzhiyao/projects/interp/bench/runs/nosol"
rows = [json.loads(l) for l in open(f"{S}/variants.jsonl")]
ok = [r for r in rows if r.get("ok") and r.get("problem")]
with open(f"{S}/inputs/variants.jsonl", "w") as o:
    for r in ok:
        o.write(json.dumps({"id": f"{r['id']}|{r['family']}|{r['j']}", "pid": r["id"], "family": r["family"],
                            "problem": r["problem"], "answer": "NaN"}, ensure_ascii=False) + "\n")
print(f"round: {len(ok)}/{len(rows)} variants accepted", flush=True)
EOF
  run qwen3.5-4b http://172.18.71.190:31201/v1 256 100000 &
  run skywork-or1-7b http://172.18.71.190:31202/v1 256 100000 &
  run dsr1-qwen3-8b http://172.18.71.190:31203/v1 512 100000 &
  run olmo3-7b-think http://172.18.47.100:31204/v1 1024 60000 &
  wait
  [ "$gen_done" = 1 ] && break
  sleep 120
done
echo "$(date '+%F %T') sampling done"
