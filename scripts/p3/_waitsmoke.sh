#!/bin/bash
# Wait for smoke to finish (writes smoke_results.json). Usage: _waitsmoke.sh [iters]
N="${1:-40}"
af=/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/artifacts/smoke_results.json
for i in $(seq 1 "$N"); do
  if [ -f "$af" ]; then
    echo "SMOKE_DONE iter=$i"
    cat "$af"
    exit 0
  fi
  if ! pgrep -f 'p3_h_accum_ceiling.py smoke' >/dev/null; then
    echo "SMOKE_EXITED_NO_RESULT iter=$i"
    exit 1
  fi
  sleep 15
done
echo "SMOKE_TIMEOUT ${N}iters"
tail -30 /tmp/p3h_smoke.log