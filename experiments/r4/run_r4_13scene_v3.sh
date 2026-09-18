#!/bin/bash
# R4 13-Scene Training Launcher (v3) — Uses r4_train_wrapper.py
#
# Runs baseline and Candidate C training for all 13 scenes.
# Scenes with existing 30K baselines only need candidate_c.
# Each GPU runs one scene at a time: baseline (if needed) then candidate_c.
#
# 8 GPUs × 13 scenes:
#   Batch 1: 8 scenes on GPUs 0-7
#   Batch 2: 5 scenes on GPUs 0-4
#
# CPU workers limited to 4 per process to avoid sshd starvation.
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

# All 13 scenes
SCENES=(bicycle bonsai counter flowers garden kitchen room stump treehill train truck drjohnson playroom)

# Scenes with existing 30K baseline checkpoints (skip baseline training)
HAS_BASELINE=(room bicycle garden)

has_baseline() {
  local s=$1
  for h in "${HAS_BASELINE[@]}"; do
    if [ "$s" = "$h" ]; then return 0; fi
  done
  return 1
}

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
  
  # Run baseline if no existing checkpoint
  if ! has_baseline "$scene"; then
    echo "[$(date)] GPU $gpu: Starting $scene BASELINE"
    CUDA_VISIBLE_DEVICES=$gpu \
    "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
      --scene "$scene" \
      --mode baseline \
      --iterations 30000 \
      --output "$baseline_dir" \
      > "$baseline_log" 2>&1
    echo "[$(date)] GPU $gpu: $scene BASELINE done (exit=$?)"
  else
    echo "[$(date)] GPU $gpu: $scene has existing baseline, skipping"
  fi
  
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

# === Batch 1: scenes 0-7 on GPUs 0-7 ===
echo "=== Batch 1: 8 scenes on 8 GPUs ==="
echo "[$(date)] Starting batch 1"
for i in 0 1 2 3 4 5 6 7; do
  run_scene_on_gpu $i $i &
done
wait
echo "=== Batch 1 complete ==="

# === Batch 2: scenes 8-12 on GPUs 0-4 ===
echo "=== Batch 2: 5 scenes on 5 GPUs ==="
echo "[$(date)] Starting batch 2"
for i in 8 9 10 11 12; do
  gpu=$((i - 8))
  run_scene_on_gpu $i $gpu &
done
wait
echo "=== Batch 2 complete ==="

echo "=== All 13 scenes complete ==="
echo "[$(date)] Done"
