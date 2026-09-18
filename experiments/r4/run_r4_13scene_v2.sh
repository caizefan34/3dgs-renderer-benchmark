#!/bin/bash
# R4 13-Scene Training Launcher (v2)
# 
# Runs baseline (MODE0) and Candidate C (MODE2) training for all 13 scenes.
# Each GPU runs one scene at a time: baseline first, then candidate_c.
# 
# 8 GPUs × 13 scenes = 2 batches:
#   Batch 1: scenes 0-7 on GPUs 0-7
#   Batch 2: scenes 8-12 on GPUs 0-4
# Each scene runs baseline then candidate_c sequentially.
#
# CPU workers limited to 4 per process to avoid sshd starvation.
#
# Usage: nohup bash experiments/r4/run_r4_13scene_v2.sh &
set -uo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PATH="$HOME/miniforge3/envs/anysplat/bin:$PATH"

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2"
LOG_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs"

mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

# Scene definitions: name|cameras_json|images_dir|sfm_ply
declare -a SCENES=(
  "bicycle|$REPO/data/official/mipnerf360/bicycle/cameras.json|$REPO/data/datasets/mipnerf360/bicycle/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/sparse/0/points3D.ply"
  "bonsai|$REPO/data/official/mipnerf360/bonsai/cameras.json|$REPO/data/datasets/mipnerf360/bonsai/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bonsai/sparse/0/points3D.ply"
  "counter|$REPO/data/official/mipnerf360/counter/cameras.json|$REPO/data/datasets/mipnerf360/counter/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/counter/sparse/0/points3D.ply"
  "flowers|$REPO/data/official/mipnerf360/flowers/cameras.json|$REPO/data/datasets/mipnerf360/flowers/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/flowers/sparse/0/points3D.ply"
  "garden|$REPO/data/official/mipnerf360/garden/cameras.json|$REPO/data/datasets/mipnerf360/garden/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/garden/sparse/0/points3D.ply"
  "kitchen|$REPO/data/official/mipnerf360/kitchen/cameras.json|$REPO/data/datasets/mipnerf360/kitchen/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/kitchen/sparse/0/points3D.ply"
  "room|$REPO/data/official/mipnerf360/room/cameras.json|$REPO/data/datasets/mipnerf360/room/images|$REPO/data/official/mipnerf360/room/point_cloud.ply"
  "stump|$REPO/data/official/mipnerf360/stump/cameras.json|$REPO/data/datasets/mipnerf360/stump/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/stump/sparse/0/points3D.ply"
  "treehill|$REPO/data/official/mipnerf360/treehill/cameras.json|$REPO/data/datasets/mipnerf360/treehill/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/treehill/sparse/0/points3D.ply"
  "train|$REPO/data/official/mipnerf360/train/cameras.json|$REPO/data/datasets/mipnerf360/train/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/tanksandtemples/train/sparse/0/points3D.ply"
  "truck|$REPO/data/official/mipnerf360/truck/cameras.json|$REPO/data/datasets/mipnerf360/truck/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/tanksandtemples/truck/sparse/0/points3D.ply"
  "drjohnson|$REPO/data/official/mipnerf360/drjohnson/cameras.json|$REPO/data/datasets/mipnerf360/drjohnson/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/deepblending/drjohnson/sparse/0/points3D.ply"
  "playroom|$REPO/data/official/mipnerf360/playroom/cameras.json|$REPO/data/datasets/mipnerf360/playroom/images|/mnt/storage_pool/liaoyuanjun/benchmark_data/deepblending/playroom/sparse/0/points3D.ply"
)

run_scene_on_gpu() {
  local idx=$1
  local gpu=$2
  local entry="${SCENES[$idx]}"
  
  IFS='|' read -r scene cameras_json images_dir sfm_ply <<< "$entry"
  
  local scene_dir="$OUTPUT_BASE/$scene"
  local baseline_dir="$scene_dir/baseline"
  local candidate_dir="$scene_dir/candidate_c"
  local baseline_log="$LOG_BASE/${scene}_baseline.log"
  local candidate_log="$LOG_BASE/${scene}_candidate.log"
  
  mkdir -p "$baseline_dir" "$candidate_dir"
  
  echo "[$(date)] GPU $gpu: Starting $scene BASELINE"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_scene.py" \
    --scene "$scene" \
    --cameras-json "$cameras_json" \
    --images-dir "$images_dir" \
    --sfm-ply "$sfm_ply" \
    --output "$baseline_dir" \
    --mode baseline \
    --iterations 30000 \
    > "$baseline_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene BASELINE done (exit=$?)"
  
  echo "[$(date)] GPU $gpu: Starting $scene CANDIDATE_C"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_scene.py" \
    --scene "$scene" \
    --cameras-json "$cameras_json" \
    --images-dir "$images_dir" \
    --sfm-ply "$sfm_ply" \
    --output "$candidate_dir" \
    --mode candidate_c \
    --iterations 30000 \
    --budget 0.05 \
    > "$candidate_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene CANDIDATE_C done (exit=$?)"
}

# === Batch 1: scenes 0-7 on GPUs 0-7 ===
echo "=== Batch 1: 8 scenes on 8 GPUs ==="
for i in 0 1 2 3 4 5 6 7; do
  run_scene_on_gpu $i $i &
done
wait
echo "=== Batch 1 complete ==="

# === Batch 2: scenes 8-12 on GPUs 0-4 ===
echo "=== Batch 2: 5 scenes on 5 GPUs ==="
for i in 8 9 10 11 12; do
  gpu=$((i - 8))
  run_scene_on_gpu $i $gpu &
done
wait
echo "=== Batch 2 complete ==="

echo "=== All 13 scenes complete ==="
echo "[$(date)] Done"
