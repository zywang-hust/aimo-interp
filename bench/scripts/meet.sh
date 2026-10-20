#!/usr/bin/env bash
# Stop a forward/reverse sampler pair once together they cover every (id, k) of the step's input.
# Killing the forward sample.py ends the script that runs it (pipefail; its trap stops the server) so a waiting
# after.sh chain proceeds; the reverse sampler on idx1 is killed by PID and run_reverse.sh's trap stops its server.
# Optional: after the stop, run <then...> on idx1 (e.g. the next step's reverse replica on the freed GPUs).
#   bench/scripts/meet.sh <step> <cfg> <n> <fwd-pid-idx0> <rev-pid-idx1> [then...]
set -uo pipefail
STEP=$1 CFG=$2 N=$3 FWD=$4 REV=$5
shift 5
R=/user/wangzhiyao/projects/interp
S=$R/bench/runs/$STEP
need=$(( $(wc -l <"$S/inputs/$CFG.jsonl") * N ))
while :; do
  have=$(python3 -I -c "
import json, os, sys
s = set()
for f in sys.argv[1:]:
    if os.path.exists(f):
        for l in open(f):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if 'answer' in r:
                s.add((r['id'], r['k']))
print(len(s))" "$S/out/$CFG.jsonl" "$S/out/$CFG.rev.jsonl")
  if [ "$have" -ge "$need" ]; then
    echo "$(date '+%F %T') $CFG: $have/$need covered; stopping fwd $FWD and rev $REV"
    /user/wangzhiyao/tools/multinode.sh on 1 "kill $REV" || true
    kill "$FWD" || true
    if [ $# -gt 0 ]; then
      sleep 120  # let run_reverse.sh's trap free the GPUs
      echo "$(date '+%F %T') $CFG: idx1 <- $*"
      /user/wangzhiyao/tools/multinode.sh on 1 "$*" || true
    fi
    exit 0
  fi
  sleep 60
done
