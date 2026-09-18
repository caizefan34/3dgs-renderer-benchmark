#!/bin/bash
# Re-launch crashed candidate_c scenes: counter (GPU 1), stump (GPU 2), flowers (GPU 7)
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/miniforge3/envs/anysplat/bin:/usr/bin:/bin:/usr/local/bin:$PATH"

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2"
LOG_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs"

launch_candidate() {
  local scene=$1
  local gpu=$2
  local output_dir="$OUTPUT_BASE/$scene/candidate_c"
  local log_file="$LOG_BASE/${scene}_candidate.log"
  mkdir -p "$output_dir/checkpoints"
  
  echo "[$(date)] GPU $gpu: Starting $scene CANDIDATE_C (re-launch)"
  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" \
    --mode candidate_c \
    --budget 0.05 \
    --iterations 30000 \
    --output "$output_dir" \
    > "$log_file" 2>&1
  echo "[$(date)] GPU $gpu: $scene CANDIDATE_C done (exit=$?)"
}

# Re-launch 3 crashed scenes
launch_candidate counter 1 &
launch_candidate stump 2 &
launch_candidate flowers 7 &
wait

echo "=== Re-launched scenes complete ==="
