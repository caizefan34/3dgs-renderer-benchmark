#!/bin/bash
echo "=== Baseline log tails ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_baseline.log"
    echo "--- $scene seed=$s baseline (last 5 lines) ---"
    tail -5 "$log" 2>/dev/null
    echo ""
  done
done

echo "=== Candidate log tails ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_candidate.log"
    echo "--- $scene seed=$s candidate (last 5 lines) ---"
    tail -5 "$log" 2>/dev/null
    echo ""
  done
done

echo "=== Launcher log ==="
tail -20 ~/r5_a_launch.log 2>/dev/null
