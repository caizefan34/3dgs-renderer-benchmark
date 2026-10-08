#!/usr/bin/env bash
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
echo "=== active trainer cmdlines ==="
pgrep -af mx_h8_trainer.py
echo "=== newest training_curve.csv across runs ==="
for d in /dev/shm/c0_quality_stability/runs/*/; do
  f="$d/training_curve.csv"
  if [ -f "$f" ]; then
    echo "--- $d ---"
    head -1 "$f"
    tail -1 "$f"
  fi
done
echo "=== gpu_samples tail (contamination) ==="
for d in /dev/shm/c0_quality_stability/runs/*/; do
  f="$d/gpu_samples.csv"
  if [ -f "$f" ]; then echo "--- $d ---"; tail -1 "$f"; fi
done
