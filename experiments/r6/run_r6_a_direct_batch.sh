#!/bin/bash
# Batch runner for R6-A direct atomic instrumentation (all 9 workloads)
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
OUT=/mnt/storage_pool/liaoyuanjun/r6_profiling
CKPT_BASE=/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts

SCENES=("room" "bicycle" "garden")
STAGES=("5000" "15000" "30000")

for scene in "${SCENES[@]}"; do
  for stage in "${STAGES[@]}"; do
    CKPT="$CKPT_BASE/$scene/checkpoints/iter_${stage}.pt"
    OUTPUT="$OUT/r6_a_direct_${scene}_${stage}.json"
    if [ -f "$OUTPUT" ]; then
      echo "SKIP (exists): $scene $stage"
      continue
    fi
    echo "=== R6-A direct: $scene $stage ==="
    CUDA_VISIBLE_DEVICES=$1 timeout 300 "$PY" "$REPO/experiments/r6/r6_a_direct_atomic.py" \
      --scene "$scene" --ckpt "$CKPT" --output "$OUTPUT" \
      --n-cameras 5 --n-tiles-sample 200 2>&1 | tail -8
    echo "  RC=$?"
  done
done
echo "=== R6-A DIRECT BATCH DONE ==="
