#!/bin/bash
# Launch batch 2: 5 remaining scenes on free GPUs
# GPU 3 is free, others will free up as batch 1 candidates finish
# Run sequentially: treehill on GPU 3, then others as GPUs free up
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/miniforge3/envs/anysplat/bin:/usr/bin:/bin:/usr/local/bin:$PATH"

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
OUTPUT_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2"
LOG_BASE="/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs"

run_scene() {
  local scene=$1
  local gpu=$2
  local baseline_dir="$OUTPUT_BASE/$scene/baseline"
  local candidate_dir="$OUTPUT_BASE/$scene/candidate_c"
  local baseline_log="$LOG_BASE/${scene}_baseline.log"
  local candidate_log="$LOG_BASE/${scene}_candidate.log"
  mkdir -p "$baseline_dir/checkpoints" "$candidate_dir/checkpoints"
  
  echo "[$(date)] GPU $gpu: $scene BASELINE"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" --mode baseline --iterations 30000 --output "$baseline_dir" \
    > "$baseline_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene BASELINE done (exit=$?)"
  
  # Clean intermediate checkpoints
  find "$baseline_dir/checkpoints/" -name 'iter_*.pt' ! -name 'iter_30000.pt' -delete 2>/dev/null
  
  echo "[$(date)] GPU $gpu: $scene CANDIDATE_C"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$REPO/experiments/r4/r4_train_wrapper.py" \
    --scene "$scene" --mode candidate_c --budget 0.05 --iterations 30000 --output "$candidate_dir" \
    > "$candidate_log" 2>&1
  echo "[$(date)] GPU $gpu: $scene CANDIDATE_C done (exit=$?)"
  
  # Clean intermediate checkpoints
  find "$candidate_dir/checkpoints/" -name 'iter_*.pt' ! -name 'iter_30000.pt' -delete 2>/dev/null
}

# Run treehill on GPU 3 first (it's free now)
run_scene treehill 3 &

# Wait for treehill baseline, then it'll do candidate
# Other scenes will start as GPUs free up from batch 1

wait
echo "=== treehill complete ==="

# Now run the remaining 4 on whatever GPUs are free
run_scene train 0 &
run_scene truck 1 &
run_scene drjohnson 2 &
run_scene playroom 3 &
wait
echo "=== All batch 2 complete ==="
