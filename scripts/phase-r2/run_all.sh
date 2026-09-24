#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_BASE="$REPO_ROOT/results/reference_v1/r2"
CKPT_BASE="$REPO_ROOT/results/reference_v1/ckpt_14k/checkpoints"
CAM_SEQ="$REPO_ROOT/results/reference_v1/room_30k/camera_sequence.npy"
RUNNER="$REPO_ROOT/experiments/r2_profile_only/corrected_concentration_runner.py"

mkdir -p "$OUTPUT_BASE"

GPU=0
for START_ITER in 2000 5000 10000 14000; do
    CKPT="$CKPT_BASE/iter_${START_ITER}.pt"
    OUTPUT="$OUTPUT_BASE/window_${START_ITER}"
    LOG="$OUTPUT_BASE/window_${START_ITER}.log"
    mkdir -p "$OUTPUT"
    echo "Launching corrected_concentration window_${START_ITER} on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 -u "$RUNNER" \
        --checkpoint "$CKPT" --start-iter "$START_ITER" --n-iters 50 \
        --output "$OUTPUT" --camera-sequence "$CAM_SEQ" > "$LOG" 2>&1 &
    GPU=$((GPU + 1))
done

echo "All 4 corrected_concentration runs launched. Waiting..."
wait
echo "All corrected_concentration runs complete."

# Now run backward timing on GPU 0
echo "Running backward timing on GPU 0"
CUDA_VISIBLE_DEVICES=0 python3 -u "$REPO_ROOT/experiments/r2_profile_only/backward_timing.py" \
    > "$OUTPUT_BASE/backward_timing.log" 2>&1
echo "Backward timing complete."
