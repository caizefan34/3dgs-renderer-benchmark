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
CKPT_BASE="$REPO/results/reference_v1/room_30k/checkpoints"
CAM_SEQ="$REPO/data/camera_sequence.npy"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r3_1_full"
PINNED_COMMIT="32ab80e"

cd "$REPO"

declare -A WINDOWS
WINDOWS[5000]="$CKPT_BASE/iter_5000.pt"
WINDOWS[15000]="$CKPT_BASE/iter_15000.pt"
WINDOWS[30000]="$CKPT_BASE/iter_30000.pt"

for START_ITER in 5000 15000 30000; do
    CKPT="${WINDOWS[$START_ITER]}"
    OUTDIR="$OUTPUT_BASE/$START_ITER"
    mkdir -p "$OUTDIR"

    echo ""
    echo "=== R3.1 Window $START_ITER (30 cameras) ==="
    echo "Checkpoint: $CKPT"
    echo "Output: $OUTDIR"

    "$PYTHON" experiments/r3/r3_certificate_runner.py \
        --checkpoint "$CKPT" \
        --start-iter "$START_ITER" \
        --n-iters 30 \
        --camera-sequence "$CAM_SEQ" \
        --output "$OUTDIR" \
        --pinned-commit "$PINNED_COMMIT"

    echo "  [DONE] Window $START_ITER"
done

echo ""
echo "=== R3.1 All 90 measurements complete ==="
echo "=== Output structure ==="
find "$OUTPUT_BASE" -type f -printf '%p %s\n' | sort
