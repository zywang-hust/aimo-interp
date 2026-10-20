#!/usr/bin/env bash
# Every config on all 36 official problems (step 1 only sampled each config on its own rows), n=16, text kept.
# Gives a full problem x config matrix of measured behaviour (p, length) to test "rollout readers" on the
# hard official rows. Servers: idx3 qwen 31201 / skywork 31202 / dsr1 31203, idx2 olmo 31204 / gpt-oss 31305+31306.
#   bench/runs/train36all/run.sh
set -uo pipefail
R=/user/wangzhiyao/projects/interp
S=$R/bench/runs/train36all
mkdir -p $S/out $S/logs
export NO_PROXY='*' no_proxy='*'
run() {  # cfg model maxtok effort urls...
  local cfg=$1 model=$2 maxtok=$3 eff=$4; shift 4
  python3 -I $R/bench/scripts/sample.py --input $R/bench/train36_input.jsonl --out $S/out/$cfg.jsonl --model $model \
    --base-url "$@" --n 16 --max-tokens $maxtok --concurrency 256 --keep-text \
    ${eff:+--reasoning-effort $eff} >> $S/logs/$cfg.log 2>&1
}
OSS="http://172.18.67.148:31305/v1 http://172.18.67.148:31306/v1"
run gpt-oss-120b__low gpt-oss-120b 100000 low $OSS &
run gpt-oss-120b__medium gpt-oss-120b 100000 medium $OSS &
run gpt-oss-120b__high gpt-oss-120b 100000 high $OSS &
run qwen3.5-4b__default qwen3.5-4b 100000 "" http://172.18.71.190:31201/v1 &
run skywork-or1-7b__default skywork-or1-7b 100000 "" http://172.18.71.190:31202/v1 &
run dsr1-qwen3-8b__default dsr1-qwen3-8b 100000 "" http://172.18.71.190:31203/v1 &
run olmo3-7b-think__default olmo3-7b-think 60000 "" http://172.18.67.148:31204/v1 &
wait
echo "$(date '+%F %T') done"
