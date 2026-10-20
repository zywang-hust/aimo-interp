#!/usr/bin/env bash
# No-solution variants of the bench v1 pool, generated and judged by DeepSeek-V4.1-Flash (thinking) on idx2.
set -euo pipefail
R=/user/wangzhiyao/projects/interp
U=http://172.18.67.148:30000/v1
X='{"chat_template_kwargs":{"thinking":true}}'
python3 -I $R/bench/scripts/perturb.py --input $R/bench/pool/pool_input.jsonl --answers $R/bench/pool/pool_input.jsonl \
  --out $R/bench/runs/nosol/variants.jsonl --families expert_no_solution_num,expert_no_solution --k 2 \
  --gen-url $U --gen-model deepseek-v4.1-flash --gen-extra "$X" --gen-max-tokens 32768 \
  --judge-url $U --judge-model deepseek-v4.1-flash --judge-extra "$X" --judge-max-tokens 32768 --concurrency 256
