#!/bin/bash
# run_trace_feats.sh <gpu> <samples.jsonl> <input.jsonl> <out.npz> <per> <shard/n>  (run on the target box)
. /user/wangzhiyao/env.sh
export TMPDIR=/user/wangzhiyao/cache/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TRITON_CACHE_DIR=/local/eval-cache/triton CUDA_CACHE_PATH=/local/eval-cache/cuda \
  CUDA_VISIBLE_DEVICES=$1 EVAL_EXTRA_SITE=engine/evalrt/kernels-overlay \
  TRACE_CKPT=512,1024,1536,2048,3072,4096 TRACE_LAYERS=12,18,24
cd /user/wangzhiyao/projects/interp
exec engine/.venv-eval/bin/python -I engine/evalrt/trace_feats.py /user/wangzhiyao/models/aimo/openai/gpt-oss-120b "$2" "$3" "$4" "$5" "$6"
