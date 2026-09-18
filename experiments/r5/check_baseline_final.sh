#!/bin/bash
echo "=== Baseline 30K PSNR ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_baseline.log"
    echo -n "$scene seed=$s: "
    grep "PSNR=" "$log" 2>/dev/null | tail -1
  done
done

echo ""
echo "=== Baseline training_metrics.json saved? ==="
for s in 1 2; do
  for scene in train truck; do
    dir="/mnt/storage_pool/liaoyuanjun/r5_a/${scene}_seed${s}/baseline"
    echo -n "$scene seed=$s: "
    if [ -f "$dir/training_metrics.json" ]; then echo "SAVED"; else echo "NOT SAVED"; fi
  done
done
