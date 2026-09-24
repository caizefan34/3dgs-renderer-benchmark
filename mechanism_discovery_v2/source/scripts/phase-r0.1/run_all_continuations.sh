#!/bin/bash
# Launch 4 continuation experiments in parallel on 4 GPUs
# Each runs 200 iterations from a different checkpoint
#
# Usage: bash scripts/phase-r0.1/run_all_continuations.sh

cd /home/liaoyuanjun/3dgs-renderer-benchmark

CKPT_DIR="results/reference_v1/ckpt_14k/checkpoints"
CAM_SEQ="results/reference_v1/room_30k/camera_sequence.npy"
OUT_BASE="results/reference_v1/r0.1"
SCRIPT="baseline/r0.1/continuation_runner.py"

echo "=== Launching 4 continuation experiments ==="
echo "Started: $(date)"

# GPU 0: checkpoint 2000
CUDA_VISIBLE_DEVICES=0 python3 -u $SCRIPT \
    --checkpoint $CKPT_DIR/iter_2000.pt \
    --start-iter 2000 \
    --n-iters 200 \
    --output $OUT_BASE/window_2000 \
    --gpu 0 \
    --camera-sequence $CAM_SEQ \
    > $OUT_BASE/window_2000.log 2>&1 &
PID_0=$!

# GPU 1: checkpoint 5000
CUDA_VISIBLE_DEVICES=1 python3 -u $SCRIPT \
    --checkpoint $CKPT_DIR/iter_5000.pt \
    --start-iter 5000 \
    --n-iters 200 \
    --output $OUT_BASE/window_5000 \
    --gpu 1 \
    --camera-sequence $CAM_SEQ \
    > $OUT_BASE/window_5000.log 2>&1 &
PID_1=$!

# GPU 2: checkpoint 10000
CUDA_VISIBLE_DEVICES=2 python3 -u $SCRIPT \
    --checkpoint $CKPT_DIR/iter_10000.pt \
    --start-iter 10000 \
    --n-iters 200 \
    --output $OUT_BASE/window_10000 \
    --gpu 2 \
    --camera-sequence $CAM_SEQ \
    > $OUT_BASE/window_10000.log 2>&1 &
PID_2=$!

# GPU 3: checkpoint 14000
CUDA_VISIBLE_DEVICES=3 python3 -u $SCRIPT \
    --checkpoint $CKPT_DIR/iter_14000.pt \
    --start-iter 14000 \
    --n-iters 200 \
    --output $OUT_BASE/window_14000 \
    --gpu 3 \
    --camera-sequence $CAM_SEQ \
    > $OUT_BASE/window_14000.log 2>&1 &
PID_3=$!

echo "Launched PIDs: $PID_0 $PID_1 $PID_2 $PID_3"
echo "Waiting for all to complete..."

wait $PID_0
echo "Window 2000 done (exit: $?)"
wait $PID_1
echo "Window 5000 done (exit: $?)"
wait $PID_2
echo "Window 10000 done (exit: $?)"
wait $PID_3
echo "Window 14000 done (exit: $?)"

echo "=== All continuations complete ==="
echo "Finished: $(date)"
