#!/bin/bash
echo "=== Remaining candidates ==="
for scene in train truck playroom; do
  log="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/${scene}_candidate.log"
  echo -n "$scene: "
  grep "ITER" "$log" 2>/dev/null | tail -1 | head -c 150
  echo
done
echo "SEP"
echo "=== Running processes ==="
ps aux | grep r4_train | grep -v grep | wc -l
