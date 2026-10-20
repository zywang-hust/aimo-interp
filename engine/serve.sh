#!/usr/bin/env bash
# Launch one SGLang server in the background.
#   engine/serve.sh <model-key> <gpu-ids> <port> [extra sglang args...]
#   e.g. engine/serve.sh dsr1-qwen3-8b 0,1 30010          # DP=2, one replica per GPU
#        engine/serve.sh gpt-oss-120b 2 30020             # TP=1
#        engine/serve.sh gpt-oss-120b 2,3 30020           # TP=2
#        engine/serve.sh qwen3.5-4b-dflash 0 30030        # with DFlash drafter
# Model keys and their settings: engine/models.yaml. Stop with engine/stop.sh <port>.
# Env knobs: MEM_FRAC (default 0.88), WAIT=0 to return without waiting for readiness.
# The card-occupier holds ~110 GiB per card; serve.sh runs tools/train/stop.sh --yield on this
# host first (yield only, never the plain stop.sh).
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,11p "$0"; exit 2; }
KEY=$1 GPUS=$2 PORT=$3; shift 3
. /user/wangzhiyao/projects/interp/engine/env.sh
cd "$ENGINE_DIR"
# engine/ is on the shared disk: pid files and logs are per host, so stop.sh on one box never
# reads another box's pids.
HOST=$(hostname); RUN=run/$HOST; LOGDIR=$ENGINE_DIR/logs/$HOST
mkdir -p "$RUN" "$LOGDIR"

if [ -f "$RUN/$PORT.pid" ] && kill -0 "$(cat "$RUN/$PORT.pid")" 2>/dev/null; then
    echo "port $PORT already served by pid $(cat "$RUN/$PORT.pid") ($(cat "$RUN/$PORT.key" 2>/dev/null))"; exit 1
fi

NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
mapfile -t ARGS < <(python - "$KEY" "$NGPU" "$PORT" <<'EOF'
import sys, yaml
key, ngpu, port = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(open("/user/wangzhiyao/projects/interp/engine/models.yaml"))
models, d = cfg["models"], cfg["defaults"]
if key not in models:
    sys.exit(f"unknown model key {key!r}; known: {', '.join(models)}")
m = models[key]
spec = m.get("spec_args", [])
if "base" in m:
    m = {**models[m["base"]], **{k: v for k, v in m.items() if k != "base"}}
a = ["--model-path", m["path"], "--served-model-name", key,
     "--host", "0.0.0.0", "--port", port,
     "--context-length", str(m.get("context_length", d["context_length"])),
     "--sampling-defaults", "openai",
     "--reasoning-parser", m["reasoning_parser"],
     "--trust-remote-code"]
if ngpu > 1:
    a += ["--dp-size" if m["parallel"] == "dp" else "--tp-size", str(ngpu)]
a += [str(x) for x in m.get("server_args", [])] + [str(x) for x in spec]
print("\n".join(a))
EOF
)
[ ${#ARGS[@]} -gt 0 ] || exit 1

# Free the cards from the occupier on THIS host only (yield, not stop).
/user/wangzhiyao/tools/train/stop.sh --yield >/dev/null 2>&1 || true
sleep 3

LOG="$LOGDIR/${KEY}-${PORT}.log"
echo "[$(date '+%F %T')] $(hostname) CUDA_VISIBLE_DEVICES=$GPUS sglang.launch_server ${ARGS[*]} --mem-fraction-static ${MEM_FRAC:-0.88} $*" >> "$LOG"
CUDA_VISIBLE_DEVICES=$GPUS setsid nohup python -m sglang.launch_server "${ARGS[@]}" \
    --mem-fraction-static "${MEM_FRAC:-0.88}" "$@" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUN/$PORT.pid"; echo "$KEY" > "$RUN/$PORT.key"
echo "started $KEY on GPUs $GPUS port $PORT pid $PID, log $LOG"

[ "${WAIT:-1}" = 0 ] && exit 0
for i in $(seq 1 360); do
    if ! kill -0 "$PID" 2>/dev/null; then echo "server died; tail of log:"; tail -30 "$LOG"; rm -f "$RUN/$PORT.pid"; exit 1; fi
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
       && curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "ready after $((i*5)) s: http://$(getent hosts "$(hostname)" | awk '{print $1}'):$PORT/v1  (model=$KEY)"; exit 0
    fi
    sleep 5
done
echo "not ready after 30 min; still running, see $LOG"; exit 1
