#!/bin/bash
# Run the opportunity probe for all three scenes sequentially on GPU 6.
set -u
GPUIDX="${1:-6}"
W=/dev/shm/bes_opp
cd "$W" || exit 1
for SC in room bicycle garden; do
  echo "=== CHAIN_PROBE scene=$SC start $(date -u +%H:%M:%S) ==="
  bash run_opp.sh "$SC" "$GPUIDX"
  echo "=== CHAIN_PROBE scene=$SC end rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== CHAIN_PROBE ALL DONE $(date -u +%H:%M:%S) ==="
