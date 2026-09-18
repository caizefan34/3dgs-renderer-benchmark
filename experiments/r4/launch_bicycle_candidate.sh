#!/bin/bash
# Launch bicycle candidate_c 30K training on GPU 5
export PATH="$HOME/miniforge3/envs/anysplat/bin:/usr/bin:/bin:/usr/local/bin:$PATH"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$HOME/3dgs-renderer-benchmark"

OUTPUT="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c"
mkdir -p "$OUTPUT"

echo "[$(date)] Starting bicycle candidate_c 30K training on GPU 5"

~/miniforge3/envs/anysplat/bin/python experiments/r4/r4_train_wrapper.py \
  --scene bicycle \
  --mode candidate_c \
  --budget 0.05 \
  --iterations 30000 \
  --output "$OUTPUT" \
  2>&1 | tee -a "$OUTPUT/train.log"

echo "[$(date)] Training finished"
