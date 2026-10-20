#!/usr/bin/env bash
# Extra gpt-oss medium samples on the 36 official problems, capped at 4096 (features only use min(len, T<=4096)).
# With train36all's 16 these give 64 per problem, so eval-time draws of 12 are close to independent (lbcheck, evalbench).
#   bench/runs/train36x/run.sh <base-url>...
set -uo pipefail
R=/user/wangzhiyao/projects/interp
S=$R/bench/runs/train36x
mkdir -p $S/out $S/logs
export NO_PROXY='*' no_proxy='*'
python3 -I $R/bench/scripts/sample.py --input $R/bench/train36_input.jsonl --out $S/out/gpt-oss-120b__medium.jsonl \
  --model gpt-oss-120b --base-url "$@" --n 48 --seed 1 --max-tokens 4096 --concurrency 256 --reasoning-effort medium \
  >> $S/logs/gpt-oss-120b__medium.log 2>&1
echo "$(date '+%F %T') done" >> $S/logs/gpt-oss-120b__medium.log
