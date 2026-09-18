#!/bin/bash
# R4 Remaining 10-Scene Training — Runs on free GPUs 0,1,2,4,7
# Each GPU runs baseline then candidate_c sequentially.
# GPUs 3,5,6 are already running room/bicycle/garden candidate_c.
set -uo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/miniforge3/envs/anysplat/bin:/usr/bin:/bin:/usr/local/bin:$PATH"

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2"
LOG_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs"
mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

# 10 remaining scenes (room, bicycle, garden already running)
SCENES=(bonsai counter flowers kitchen stump treehill train truck drjohnson playroom)
GPUS=(0 1 2 4 7)  # 5 free GPUs

run_scene_on_gpu() {
  local idx=$1
  local gpu=$2
  local scene="${SCENES[$idx]}"
  
  local scene_dir="$OUTPUT_BASE/$scene"
  local baseline_dir="$scene_dir/baseline"
  local candidate_dir="$scene_dir/candidate_c"
  local baseline_log="$LOG_BASE/${scene}_baseline.log"
  local candidate_log="$LOG_BASE/${scene}_candidate.log"
  
  mkdir -p "$baseline_dir" "$candidate_dir"
  
  # Run baseline
  echo "[$(date)] GPU $gpu: Starting $scene BASELINE"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" \
    --mode baseline \
    --iterations 30000 \
    --output "$baseline_dir" \
    > "$baseline_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene BASELINE done (exit=$?)"
  
  # Run candidate_c
  echo "[$(date)] GPU $gpu: Starting $scene CANDIDATE_C"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" \
    --mode candidate_c \
    --budget 0.05 \
    --iterations 30000 \
    --output "$candidate_dir" \
    > "$candidate_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene CANDIDATE_C done (exit=$?)"
}

# === Batch 1: scenes 0-4 on GPUs 0,1,2,4,7 ===
echo "=== Batch 1: 5 scenes on GPUs 0,1,2,4,7 ==="
echo "[$(date)] Starting batch 1"
for i in 0 1 2 3 4; do
  run_scene_on_gpu $i ${GPUS[$i]} &
done
wait
echo "=== Batch 1 complete ==="

# === Batch 2: scenes 5-9 on GPUs 0,1,2,4,7 ===
echo "=== Batch 2: 5 scenes on GPUs 0,1,2,4,7 ==="
echo "[$(date)] Starting batch 2"
for i in 5 6 7 8 9; do
  gpu_idx=$((i - 5))
  run_scene_on_gpu $i ${GPUS[$gpu_idx]} &
done
wait
echo "=== Batch 2 complete ==="

echo "=== All 10 remaining scenes complete ==="
echo "[$(date)] Done"
