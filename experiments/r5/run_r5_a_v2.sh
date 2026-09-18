#!/bin/bash
# R5-A v2: Re-launch all 8 new runs (no checkpoint saving)
# Baselines crashed at 30K checkpoint save, candidates killed early
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
  mkdir -p "$b_dir" "$c_dir"
  
  # Baseline
  echo "[$(date)] GPU $gpu: $scene seed=$seed BASELINE"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" --mode baseline --iterations 30000 \
    --seed "$seed" --output "$b_dir" \
    > "$b_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene seed=$seed BASELINE done (exit=$?)"
  
  # Candidate C
  echo "[$(date)] GPU $gpu: $scene seed=$seed CANDIDATE_C"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" --mode candidate_c --budget 0.05 --iterations 30000 \
    --seed "$seed" --output "$c_dir" \
    > "$c_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene seed=$seed CANDIDATE_C done (exit=$?)"
}

# 4 GPUs, each runs B then C sequentially
run_pair train 1 0 &
run_pair truck 1 1 &
run_pair train 2 2 &
run_pair truck 2 3 &

wait
echo "=== All R5-A v2 runs complete ==="
echo "[$(date)] Done"
