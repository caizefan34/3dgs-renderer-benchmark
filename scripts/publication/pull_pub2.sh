#!/bin/bash
# Pull publication run artifacts (B1 gate, P4 corners) into a tar on mx
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/pub_pull2 /tmp/pub_pull2.tar
mkdir -p /tmp/pub_pull2
for d in b1_room b1_bicycle b1_garden b1ae03_room b1ae03_bicycle b1ae03_garden c0e01_bicycle c0e01_garden; do
  mkdir -p /tmp/pub_pull2/$d
  for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json memory.json; do
    [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_pull2/$d/
  done
done
cp b1_gate_verdict.json /tmp/pub_pull2/ 2>/dev/null || true
tar cf /tmp/pub_pull2.tar -C /tmp/pub_pull2 .
echo "PACKED: $(find /tmp/pub_pull2 -type f | wc -l) files"
