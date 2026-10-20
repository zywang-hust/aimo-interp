#!/usr/bin/env bash
# Task B: load + forward timing for the 4 small models in the eval-runtime replica, one GPU.
. /user/wangzhiyao/env.sh
export TMPDIR=/user/wangzhiyao/cache/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/local/eval-cache/triton TORCHINDUCTOR_CACHE_DIR=/local/eval-cache/inductor CUDA_CACHE_PATH=/local/eval-cache/cuda
unset HF_HUB_ENABLE_HF_TRANSFER
export CUDA_VISIBLE_DEVICES=${GPU:-3}
cd /user/wangzhiyao/projects/interp/engine
R=/user/wangzhiyao/models/aimo
for m in "$@"; do
  extra=""; [[ $m == *DeepSeek-R1-0528* ]] && extra="--fix-tokenizer"
  echo "### $m $(date +%T)"
  .venv-eval/bin/python -I eval_check.py $R/$m $extra 2>&1 | grep -vE "FutureWarning|warnings.warn"
done
