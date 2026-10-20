#!/usr/bin/env bash
# Second replica for a slow config: serve <key> on <gpus>:<port> and sample <step>'s input from the end into
# <cfg>.rev.jsonl, skipping pairs the forward run has already done. Stop it (stop.sh <port>) when they meet.
#   bench/scripts/run_reverse.sh <model-key> <gpu-ids> <port> <step> [n=8]
set -euo pipefail
KEY=$1 GPUS=$2 PORT=$3 STEP=$4 N=${5:-8}
R=/user/wangzhiyao/projects/interp
S=$R/bench/runs/$STEP
MAXTOK=100000
[ "$KEY" = olmo3-7b-think ] && MAXTOK=60000
mkdir -p "$S/logs"
bash "$R/engine/serve.sh" "$KEY" "$GPUS" "$PORT"
trap 'bash "$R/engine/stop.sh" "$PORT" || true' EXIT
python3 -I "$R/bench/scripts/sample.py" --input "$S/inputs/${KEY}__default.jsonl" --out "$S/out/${KEY}__default.rev.jsonl" \
  --skip-done-in "$S/out/${KEY}__default.jsonl" --reverse --model "$KEY" --base-url "http://127.0.0.1:$PORT/v1" \
  --n "$N" --max-tokens "$MAXTOK" --concurrency 256 2>&1 | tee -a "$S/logs/${KEY}__default.rev.log"
