#!/bin/bash
# R0.2: Launch 4 continuation experiments in parallel on GPUs 0-3
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_BASE="$REPO_ROOT/results/reference_v1/r0.2"
CKPT_BASE="$REPO_ROOT/results/reference_v1/ckpt_14k/checkpoints"
CAM_SEQ="$REPO_ROOT/results/reference_v1/room_30k/camera_sequence.npy"
RUNNER="$REPO_ROOT/baseline/r0.2/continuation_runner.py"

mkdir -p "$OUTPUT_BASE"

# Window configurations: start_iter checkpoint
declare -A WINDOWS
WINDOWS[2000]="2000"
WINDOWS[5000]="5000"
WINDOWS[10000]="10000"
WINDOWS[14000]="14000"

GPU=0
for START_ITER in 2000 5000 10000 14000; do
    CKPT="$CKPT_BASE/iter_${START_ITER}.pt"
    OUTPUT="$OUTPUT_BASE/window_${START_ITER}"
    LOG="$OUTPUT_BASE/window_${START_ITER}.log"

    mkdir -p "$OUTPUT"

    echo "Launching window_${START_ITER} on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 -u "$RUNNER" \
        --checkpoint "$CKPT" \
        --start-iter "$START_ITER" \
        --n-iters 200 \
        --output "$OUTPUT" \
        --gpu "$GPU" \
        --camera-sequence "$CAM_SEQ" \
        > "$LOG" 2>&1 &

    GPU=$((GPU + 1))
done

echo "All 4 continuations launched. Waiting..."
wait
echo "All continuations complete."
