#!/bin/bash
# R6-4 warp duplicate batch runner on GPU 5
set -uo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniforge3/envs/anysplat/bin/python
REPO=~/3dgs-renderer-benchmark
OUT=/mnt/storage_pool/liaoyuanjun/r6_profiling
CKPT_BASE=/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts

run_warp() {
  local scene=$1 stage=$2
  local ckpt="$CKPT_BASE/$scene/checkpoints/iter_${stage}.pt"
  if [ ! -f "$ckpt" ]; then echo "SKIP $scene $stage"; return; fi
  local outfile="$OUT/r6_4_${scene}_${stage}.json"
  if [ -f "$outfile" ]; then echo "EXISTS $scene $stage"; return; fi
  echo "=== Warp dup $scene @ ${stage} ==="
  CUDA_VISIBLE_DEVICES=5 timeout 120 "$PY" "$REPO/experiments/r6/r6_4_warp_duplicate.py" \
    --scene "$scene" --ckpt "$ckpt" --output "$outfile" --n-cameras 10 2>&1 | tail -5
}

for scene in room bicycle garden; do
  for stage in 5000 15000 30000; do
    run_warp $scene $stage
  done
done
echo "=== WARP DUP DONE ==="
