#!/bin/bash
# R5-A: 8 new paired runs for seeds 1 and 2
# train and truck, baseline and candidate_c
# 8 runs on 8 GPUs in parallel (Wave 1)
set -uo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/miniforge3/envs/anysplat/bin:/usr/bin:/bin:/usr/local/bin:$PATH"

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r5_a"
LOG_BASE="/mnt/storage_pool/liaoyuanjun/r5_a_logs"
mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

run_pair() {
  local scene=$1
  local seed=$2
  local gpu=$3
  
  local b_dir="$OUTPUT_BASE/${scene}_seed${seed}/baseline"
  local c_dir="$OUTPUT_BASE/${scene}_seed${seed}/candidate_c"
  local b_log="$LOG_BASE/${scene}_seed${seed}_baseline.log"
  local c_log="$LOG_BASE/${scene}_seed${seed}_candidate.log"
  mkdir -p "$b_dir/checkpoints" "$c_dir/checkpoints"
  
  # Baseline
  echo "[$(date)] GPU $gpu: $scene seed=$seed BASELINE"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" --mode baseline --iterations 30000 \
    --seed "$seed" --output "$b_dir" \
    > "$b_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene seed=$seed BASELINE done (exit=$?)"
  
  # Clean intermediate checkpoints
  find "$b_dir/checkpoints/" -name 'iter_*.pt' ! -name 'iter_30000.pt' -delete 2>/dev/null
  
  # Candidate C
  echo "[$(date)] GPU $gpu: $scene seed=$seed CANDIDATE_C"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" --mode candidate_c --budget 0.05 --iterations 30000 \
    --seed "$seed" --output "$c_dir" \
    > "$c_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene seed=$seed CANDIDATE_C done (exit=$?)"
  
  # Clean intermediate checkpoints
  find "$c_dir/checkpoints/" -name 'iter_*.pt' ! -name 'iter_30000.pt' -delete 2>/dev/null
}

# 8 runs: 2 scenes × 2 seeds × 2 methods (sequential per GPU)
# Assign to 8 GPUs, each GPU runs B then C sequentially

# GPU 0: train seed=1
run_pair train 1 0 &
# GPU 1: truck seed=1
run_pair truck 1 1 &
# GPU 2: train seed=2
run_pair train 2 2 &
# GPU 3: truck seed=2
run_pair truck 2 3 &

wait
echo "=== All R5-A runs complete ==="
echo "[$(date)] Done"
