#!/bin/bash
# Quick test: Run room with 100 iterations in both modes
export PATH="$HOME/miniforge3/envs/anysplat/bin:$PATH"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

cd "$HOME/3dgs-renderer-benchmark"

echo "=== Quick test: room baseline 100 iters ==="
timeout 300 "$HOME/miniforge3/envs/anysplat/bin/python" experiments/r4/r4_train_wrapper.py \
  --scene room \
  --mode baseline \
  --iterations 100 \
  --output /mnt/storage_pool/liaoyuanjun/r4_test/room_baseline \
  2>&1 | tail -20

echo ""
echo "=== Quick test: room candidate_c 100 iters ==="
timeout 300 "$HOME/miniforge3/envs/anysplat/bin/python" experiments/r4/r4_train_wrapper.py \
  --scene room \
  --mode candidate_c \
  --iterations 100 \
  --budget 0.05 \
  --output /mnt/storage_pool/liaoyuanjun/r4_test/room_candidate \
  2>&1 | tail -20

echo "=== Quick test done ==="
