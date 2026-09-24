#!/bin/bash
# R3.1 Parallel Run — 3 windows on 3 GPUs simultaneously
# Each window: 30 cameras on a dedicated A100
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
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r3_1_full"
PINNED_COMMIT="32ab80e"

cd "$REPO"

run_window() {
    local START_ITER=$1
    local GPU=$2
    local CKPT=$3
    local OUTDIR="$OUTPUT_BASE/$START_ITER"
    mkdir -p "$OUTDIR"

    echo "[$START_ITER] Starting on GPU $GPU, checkpoint $CKPT"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" experiments/r3/r3_certificate_runner.py \
        --checkpoint "$CKPT" \
        --start-iter "$START_ITER" \
        --n-iters 30 \
        --camera-sequence "$CAM_SEQ" \
        --output "$OUTDIR" \
        --pinned-commit "$PINNED_COMMIT" \
        > "$OUTDIR/runner.log" 2>&1
    echo "[$START_ITER] DONE"
}

# Launch all 3 windows in parallel on GPU 0, 1, 2
run_window 5000  0 "$CKPT_BASE/iter_5000.pt"  &
PID_5K=$!
run_window 15000 1 "$CKPT_BASE/iter_15000.pt" &
PID_15K=$!
run_window 30000 2 "$CKPT_BASE/iter_30000.pt" &
PID_30K=$!

echo "Launched: 5K=PID_$PID_5K, 15K=PID_$PID_15K, 30K=PID_$PID_30K"
echo "Waiting for all windows..."

wait $PID_5K  && echo "5K window complete" || echo "5K window FAILED"
wait $PID_15K && echo "15K window complete" || echo "15K window FAILED"
wait $PID_30K && echo "30K window complete" || echo "30K window FAILED"

echo ""
echo "=== R3.1 All 90 measurements complete ==="
echo "=== Output structure ==="
find "$OUTPUT_BASE" -type f -printf '%p %s\n' | sort
