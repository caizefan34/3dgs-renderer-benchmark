#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR="/mnt/storage_pool/liaoyuanjun/torch_ext_r3_1"
export CUDA_CACHE_PATH="/mnt/storage_pool/liaoyuanjun/cuda_cache_r3_1"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
CKPT="$REPO/results/reference_v1/room_30k/checkpoints/iter_5000.pt"
CAM_SEQ="$REPO/data/camera_sequence.npy"
OUTDIR="/mnt/storage_pool/liaoyuanjun/r3_1_smoke"
PINNED_COMMIT="32ab80e"

cd "$REPO"

echo "=== R3.1 Smoke Test (1 camera, 5K snapshot) ==="
echo "Python: $PYTHON"
echo "Checkpoint: $CKPT"
echo "Output: $OUTDIR"

"$PYTHON" experiments/r3/r3_certificate_runner.py \
    --checkpoint "$CKPT" \
    --start-iter 5000 \
    --n-iters 1 \
    --camera-sequence "$CAM_SEQ" \
    --output "$OUTDIR" \
    --pinned-commit "$PINNED_COMMIT"

echo "=== Smoke test complete ==="
echo "=== Output files ==="
ls -la "$OUTDIR/"
