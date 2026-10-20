#!/usr/bin/env bash
# Run a python script in the eval-runtime replica on one GPU:  GPU=<id> engine/evalrt/run_eval.sh <script.py> [args]
. /user/wangzhiyao/env.sh
export TMPDIR=/user/wangzhiyao/cache/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/local/eval-cache/triton TORCHINDUCTOR_CACHE_DIR=/local/eval-cache/inductor CUDA_CACHE_PATH=/local/eval-cache/cuda
export EVAL_EXTRA_SITE=engine/evalrt/kernels-overlay
unset HF_HUB_ENABLE_HF_TRANSFER
export CUDA_VISIBLE_DEVICES=${GPU:-3}
cd /user/wangzhiyao/projects/interp
engine/.venv-eval/bin/python -I "$@" 2>&1 | grep --line-buffered -vE "FutureWarning|warnings.warn|Loading weights|_warn_once|USDT:|Skipping publisher|return get_kernel_hub"
