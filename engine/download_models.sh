#!/usr/bin/env bash
# Pull the evaluated models + DFlash drafters from ModelScope (direct, no proxy, multi-stream).
# Everything stays on the persistent disk; never under /root.
set -u
. /user/wangzhiyao/env.sh
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export MODELSCOPE_CACHE=/user/wangzhiyao/cache/modelscope TMPDIR=/user/wangzhiyao/cache/tmp
B=/user/wangzhiyao/models/aimo
L=/user/wangzhiyao/logs/aimo-dl
mkdir -p "$B" "$L" "$MODELSCOPE_CACHE" "$TMPDIR"

for m in Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think \
         deepseek-ai/DeepSeek-R1-0528-Qwen3-8B z-lab/gpt-oss-120b-DFlash z-lab/Qwen3.5-4B-DFlash; do
  modelscope download --model "$m" --local-dir "$B/$m" --max-workers 8 > "$L/${m//\//__}.log" 2>&1 &
done
# HF-format weights only (matches HF revision b5c939de); skip original/ and metal/
modelscope download --model openai-mirror/gpt-oss-120b --local-dir "$B/openai/gpt-oss-120b" \
  --exclude 'original/*' 'metal/*' --max-workers 14 > "$L/openai__gpt-oss-120b.log" 2>&1 &
wait
echo "all downloads finished"
du -sh "$B"/*/*
