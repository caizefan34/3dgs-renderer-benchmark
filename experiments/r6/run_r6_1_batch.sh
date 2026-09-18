#!/bin/bash
# R6-1 batch runner: profile all available checkpoints on GPU 4
set -uo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniforge3/envs/anysplat/bin/python
REPO=~/3dgs-renderer-benchmark
OUT=/mnt/storage_pool/liaoyuanjun/r6_profiling
CKPT_BASE=/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts
mkdir -p "$OUT"

run_profile() {
  local scene=$1 stage=$2
  local ckpt="$CKPT_BASE/$scene/checkpoints/iter_${stage}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "SKIP $scene $stage (no checkpoint)"
    return
  fi
  local outfile="$OUT/r6_1_${scene}_${stage}.json"
  if [ -f "$outfile" ]; then
    echo "EXISTS $scene $stage"
    return
  fi
  echo "=== Profiling $scene @ ${stage} ==="
  CUDA_VISIBLE_DEVICES=4 timeout 300 "$PY" "$REPO/experiments/r6/r6_1_bwd_decompose.py" \
    --scene "$scene" --ckpt "$ckpt" --output "$outfile" \
    --n-warmup 20 --n-measure 100 2>&1 | tail -15
  echo "RC=$?"
}

# Run all available checkpoints
for scene in room bicycle garden; do
  for stage in 5000 15000 30000; do
    run_profile $scene $stage
  done
done

echo "=== ALL DONE ==="
ls -la "$OUT"/r6_1_*.json 2>/dev/null
