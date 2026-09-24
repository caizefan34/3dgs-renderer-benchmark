#!/bin/bash
# Launch ONE window of R3 on the A100 with a fresh clean environment
set -euo pipefail

cd /home/liaoyuanjun/3dgs-renderer-benchmark
pkill -f r3_certificate_runner 2>/dev/null || true
sleep 1

rm -rf /tmp/torch_extensions_r3 /tmp/cuda_cache_r3
mkdir -p /tmp/torch_extensions_r3 /tmp/cuda_cache_r3
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_r3
export CUDA_CACHE_PATH=/tmp/cuda_cache_r3
export CUDA_VISIBLE_DEVICES=1

OUTDIR="results/reference_v1/r3/5000"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

python3 -u experiments/r3/r3_certificate_runner.py \
    --checkpoint "results/reference_v1/room_30k/checkpoints/iter_5000.pt" \
    --start-iter 5000 \
    --n-iters 30 \
    --camera-sequence "data/camera_sequence.npy" \
    --output "$OUTDIR" \
    --pinned-commit e494458 || true

echo "=====FINAL_EXIT=$?====="
ls -la "$OUTDIR"
