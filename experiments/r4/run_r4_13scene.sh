#!/bin/bash
# R4 13-Scene Parallel Training Launcher
# 
# Runs baseline (MODE0) and Candidate C (MODE2) training for all 13 scenes.
# Each scene gets one GPU. 8 GPUs → 2 batches.
#
# Usage: bash experiments/r4/run_r4_13scene.sh
#
# IMPORTANT: Limit CPU workers to avoid sshd starvation

set -uo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene"
LOG_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_logs"

mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

# Scene → data path mapping
# Format: scene_name|data_dir|sfm_ply|dataset_type
declare -a SCENES=(
  "bicycle|$REPO/data/official/mipnerf360/bicycle|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/sparse/0/points3D.ply|mipnerf360"
  "bonsai|$REPO/data/official/mipnerf360/bonsai|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bonsai/sparse/0/points3D.ply|mipnerf360"
  "counter|$REPO/data/official/mipnerf360/counter|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/counter/sparse/0/points3D.ply|mipnerf360"
  "flowers|$REPO/data/official/mipnerf360/flowers|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/flowers/sparse/0/points3D.ply|mipnerf360"
  "garden|$REPO/data/official/mipnerf360/garden|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/garden/sparse/0/points3D.ply|mipnerf360"
  "kitchen|$REPO/data/official/mipnerf360/kitchen|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/kitchen/sparse/0/points3D.ply|mipnerf360"
  "room|$REPO/data/official/mipnerf360/room|$REPO/data/official/mipnerf360/room/point_cloud.ply|mipnerf360"
  "stump|$REPO/data/official/mipnerf360/stump|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/stump/sparse/0/points3D.ply|mipnerf360"
  "treehill|$REPO/data/official/mipnerf360/treehill|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/treehill/sparse/0/points3D.ply|mipnerf360"
  "train|$REPO/data/official/mipnerf360/train|/mnt/storage_pool/liaoyuanjun/benchmark_data/tanksandtemples/train/sparse/0/points3D.ply|tanksandtemples"
  "truck|$REPO/data/official/mipnerf360/truck|/mnt/storage_pool/liaoyuanjun/benchmark_data/tanksandtemples/truck/sparse/0/points3D.ply|tanksandtemples"
  "drjohnson|$REPO/data/official/mipnerf360/drjohnson|/mnt/storage_pool/liaoyuanjun/benchmark_data/deepblending/drjohnson/sparse/0/points3D.ply|deepblending"
  "playroom|$REPO/data/official/mipnerf360/playroom|/mnt/storage_pool/liaoyuanjun/benchmark_data/deepblending/playroom/sparse/0/points3D.ply|deepblending"
)

# Batch 1: scenes 0-7 (GPUs 0-7)
# Batch 2: scenes 8-12 (GPUs 0-4)

run_scene() {
  local idx=$1
  local gpu=$2
  local mode=$3  # "baseline" or "candidate_c"
  local entry="${SCENES[$idx]}"
  
  IFS='|' read -r scene data_dir sfm_ply dtype <<< "$entry"
  
  local out_dir="$OUTPUT_BASE/${scene}/${mode}"
  local log_file="$LOG_BASE/${scene}_${mode}.log"
  
  mkdir -p "$out_dir"
  
  echo "[$(date)] Starting $scene ($mode) on GPU $gpu"
  
  CUDA_VISIBLE_DEVICES=$gpu \
  OMP_NUM_THREADS=4 \
  "$PYTHON" "$REPO/experiments/r4/r4_train_scene.py" \
    --scene "$scene" \
    --data-dir "$data_dir" \
    --sfm-ply "$sfm_ply" \
    --dataset-type "$dtype" \
    --output "$out_dir" \
    --mode "$mode" \
    --iterations 30000 \
    --budget 0.05 \
    > "$log_file" 2>&1
  
  echo "[$(date)] $scene ($mode) on GPU $gpu DONE (exit=$?)" >> "$log_file"
}

# === Batch 1: scenes 0-7, each on one GPU ===
echo "=== Batch 1: 8 scenes on 8 GPUs ==="
for i in 0 1 2 3 4 5 6 7; do
  run_scene $i $i "baseline" &
  run_scene $i $i "candidate_c" &
done

# Wait for batch 1
wait
echo "=== Batch 1 complete ==="

# === Batch 2: scenes 8-12, on GPUs 0-4 ===
echo "=== Batch 2: 5 scenes on 5 GPUs ==="
for i in 8 9 10 11 12; do
  gpu=$((i - 8))
  run_scene $i $gpu "baseline" &
  run_scene $i $gpu "candidate_c" &
done

wait
echo "=== Batch 2 complete ==="
echo "=== All 13 scenes complete ==="
