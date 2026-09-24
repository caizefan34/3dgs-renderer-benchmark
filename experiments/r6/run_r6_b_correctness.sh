#!/bin/bash
# Run R6-B correctness on all 3 scenes
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
OUT=/mnt/storage_pool/liaoyuanjun/r6_profiling
CKPT_BASE=/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts

for scene in room bicycle garden; do
  CKPT="$CKPT_BASE/$scene/checkpoints/iter_30000.pt"
  OUTPUT="$OUT/r6_b_correctness_${scene}_30k.json"
  echo "=== R6-B correctness: $scene 30K ==="
  CUDA_VISIBLE_DEVICES=7 timeout 120 "$PY" "$REPO/experiments/r6/r6_b_correctness.py" \
    --scene "$scene" --ckpt "$CKPT" --output "$OUTPUT" 2>&1 | tail -20
  echo "RC=$?"
done
echo "=== R6-B CORRECTNESS DONE ==="
