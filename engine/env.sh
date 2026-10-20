# Source me: environment for the interp sampling engine (SGLang 0.5.22, torch 2.14.1+cu130).
# All paths absolute; nothing under /root ($HOME is an ephemeral overlay).
. /user/wangzhiyao/env.sh
export ENGINE_DIR=/user/wangzhiyao/projects/interp/engine
export VIRTUAL_ENV=$ENGINE_DIR/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
export TMPDIR=/user/wangzhiyao/cache/tmp
export UV_LINK_MODE=copy
export HF_HOME=/user/wangzhiyao/cache/huggingface
export XDG_CACHE_HOME=/user/wangzhiyao/cache/xdg
# JIT caches must live on /local (yrfs rename is not atomic across processes, see CLAUDE.md).
# Same path on every box; separate from the DeepSeek deployments' caches (different torch/flashinfer ABI).
export SGLANG_CACHE_DIR=/local/sglang-cache/interp-engine
export SGLANG_JIT_CACHE_DIR=$SGLANG_CACHE_DIR/jit
export FLASHINFER_WORKSPACE_BASE=$SGLANG_CACHE_DIR/flashinfer
export TRITON_CACHE_DIR=$SGLANG_CACHE_DIR/triton
export TILELANG_CACHE_DIR=$SGLANG_CACHE_DIR/tilelang
export TORCHINDUCTOR_CACHE_DIR=$SGLANG_CACHE_DIR/inductor
export CUDA_CACHE_PATH=$SGLANG_CACHE_DIR/cuda
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF HF_HUB_ENABLE_HF_TRANSFER
# Serving is local-only; keep the proxy away from localhost traffic and model loading.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
mkdir -p "$SGLANG_JIT_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TILELANG_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR" 2>/dev/null
