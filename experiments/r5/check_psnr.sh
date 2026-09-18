#!/bin/bash
echo "=== Disk space ==="
df -h /mnt/storage_pool/liaoyuanjun/ 2>/dev/null | tail -1

echo ""
echo "=== Baseline PSNR values ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_baseline.log"
    echo "--- $scene seed=$s baseline ---"
    grep "PSNR=" "$log" 2>/dev/null
    echo ""
  done
done

echo "=== Candidate PSNR values ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_candidate.log"
    echo "--- $scene seed=$s candidate ---"
    grep "PSNR=" "$log" 2>/dev/null
    echo ""
  done
done

echo "=== Candidate last lines ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_candidate.log"
    echo "--- $scene seed=$s candidate ---"
    tail -3 "$log" 2>/dev/null
    echo ""
  done
done
