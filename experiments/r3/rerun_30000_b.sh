#!/bin/bash
# Restart 30000_B on a free GPU
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
OUT="/mnt/storage_pool/liaoyuanjun/r3_1_parallel/30000_B"

cd "$REPO"

# Clean old incomplete output
rm -f "$OUT/certificate_correctness.json" "$OUT/certificate_tightness.json" "$OUT/exact_zero_statistics.json" "$OUT/complexity_accounting.json" "$OUT/certificate_disabled.json" "$OUT/tile_gaussian_certificate.json" "$OUT/pair_records.npz"

echo "=== Restarting 30000_B on GPU 0 ==="
date
"$PYTHON" experiments/r3/r3_certificate_runner.py \
  --checkpoint results/reference_v1/room_30k/checkpoints/iter_30000.pt \
  --start-iter 29980 \
  --n-iters 10 \
  --camera-sequence results/reference_v1/camera_sequence.npy \
  --output "$OUT" \
  --pinned-commit 32ab80e \
  > "$OUT/runner_rerun.log" 2>&1
echo "=== 30000_B rerun done, exit=$? ==="
date
