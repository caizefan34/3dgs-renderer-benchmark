#!/bin/bash
# R6-5 fusion oracle batch runner on GPU 5
set -uo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniforge3/envs/anysplat/bin/python
REPO=~/3dgs-renderer-benchmark
OUT=/mnt/storage_pool/liaoyuanjun/r6_profiling
CKPT_BASE=/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts

run_fusion() {
  local scene=$1 stage=$2
  local ckpt="$CKPT_BASE/$scene/checkpoints/iter_${stage}.pt"
  if [ ! -f "$ckpt" ]; then echo "SKIP $scene $stage"; return; fi
  local outfile="$OUT/r6_5_${scene}_${stage}.json"
  if [ -f "$outfile" ]; then echo "EXISTS $scene $stage"; return; fi
  echo "=== Fusion oracle $scene @ ${stage} ==="
  CUDA_VISIBLE_DEVICES=5 timeout 300 "$PY" "$REPO/experiments/r6/r6_5_fusion_oracle.py" \
    --scene "$scene" --ckpt "$ckpt" --output "$outfile" \
    --n-warmup 20 --n-measure 100 2>&1 | tail -12
}

for scene in room bicycle garden; do
  for stage in 5000 15000 30000; do
    run_fusion $scene $stage
  done
done
echo "=== FUSION ORACLE DONE ==="
