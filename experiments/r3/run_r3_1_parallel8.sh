#!/bin/bash
# R3.1 Parallel Sub-Batch Run — 8 GPUs, 9 sub-batches
# Splits each window's 30 cameras into sub-batches for parallel execution
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR="/mnt/storage_pool/liaoyuanjun/torch_ext_r3_1"
export CUDA_CACHE_PATH="/mnt/storage_pool/liaoyuanjun/cuda_cache_r3_1"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
CKPT_BASE="$REPO/results/reference_v1/room_30k/checkpoints"
CAM_SEQ="$REPO/data/camera_sequence.npy"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r3_1_parallel"
PINNED_COMMIT="32ab80e"

cd "$REPO"
mkdir -p "$OUTPUT_BASE"

run_batch() {
    local LABEL=$1
    local GPU=$2
    local CKPT=$3
    local START=$4
    local N=$5
    local OUTDIR="$OUTPUT_BASE/$LABEL"
    mkdir -p "$OUTDIR"

    echo "[$LABEL] GPU=$GPU start=$START n=$N ckpt=$CKPT"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" experiments/r3/r3_certificate_runner.py \
        --checkpoint "$CKPT" \
        --start-iter "$START" \
        --n-iters "$N" \
        --camera-sequence "$CAM_SEQ" \
        --output "$OUTDIR" \
        --pinned-commit "$PINNED_COMMIT" \
        > "$OUTDIR/runner.log" 2>&1
    echo "[$LABEL] DONE"
}

# 5K window: 3 sub-batches of 10 cameras each (GPU 0, 3, 4)
run_batch 5000_A 0 "$CKPT_BASE/iter_5000.pt"  5000 10 &
run_batch 5000_B 3 "$CKPT_BASE/iter_5000.pt"  5010 10 &
run_batch 5000_C 4 "$CKPT_BASE/iter_5000.pt"  5020 10 &

# 15K window: 3 sub-batches of 10 cameras each (GPU 1, 5, 6)
run_batch 15000_A 1 "$CKPT_BASE/iter_15000.pt" 15000 10 &
run_batch 15000_B 5 "$CKPT_BASE/iter_15000.pt" 15010 10 &
run_batch 15000_C 6 "$CKPT_BASE/iter_15000.pt" 15020 10 &

# 30K window: 3 sub-batches of 10 cameras each (GPU 2, 7, 3-wait)
# GPU 3 and 4 are shared with 5K, so use GPU 2, 7 for first two batches
# and GPU 3 for the third (after 5000_B finishes or concurrently)
run_batch 30000_A 2 "$CKPT_BASE/iter_30000.pt" 29970 10 &
run_batch 30000_B 7 "$CKPT_BASE/iter_30000.pt" 29980 10 &
run_batch 30000_C 3 "$CKPT_BASE/iter_30000.pt" 29990 10 &

echo "Launched 9 sub-batches on 8 GPUs"
echo "Waiting for all..."

wait
echo ""
echo "=== All sub-batches complete ==="
find "$OUTPUT_BASE" -name "certificate_correctness.json" -printf "%p %s\n" | sort
