#!/usr/bin/env bash
# Run a command once a pid has exited (zombies count as exited: `kill -0` succeeds on them).
#   bench/scripts/after.sh <pid> <command> [args...]
P=$1; shift
while st=$(ps -o stat= -p "$P" 2>/dev/null) && [ -n "$st" ] && [[ "$st" != Z* ]]; do sleep 30; done
exec "$@"
