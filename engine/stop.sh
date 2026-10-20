#!/usr/bin/env bash
# Stop servers started by engine/serve.sh ON THIS HOST, by PID (process group), never by pattern.
#   engine/stop.sh <port> [<port>...]     engine/stop.sh all
# Only reads run/$(hostname)/: engine/ is on the shared disk and other boxes keep their own pid files.
# The card-occupier reclaims a card only after it has been idle for 15 min; nothing to do here.
cd /user/wangzhiyao/projects/interp/engine || exit 1
RUN=run/$(hostname)
ports=("$@"); [ "${1:-}" = all ] && ports=($(ls "$RUN"/*.pid 2>/dev/null | sed "s#$RUN/##; s#\.pid##"))
for port in "${ports[@]}"; do
    f=$RUN/$port.pid; [ -f "$f" ] || { echo "$port: no pid file in $RUN"; continue; }
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"; }
        echo "$port: stopped pid $pid ($(cat "$RUN/$port.key" 2>/dev/null))"
    else
        echo "$port: pid $pid not running"
    fi
    rm -f "$f" "$RUN/$port.key"
done
