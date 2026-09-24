#!/bin/bash
set -e
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/home/liaoyuanjun/.local/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=4
export HIGS_PX_RUNTIME=2

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
SOURCE=/tmp/higs_scalar_adj_freeze_source
CORE_SO=/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so
OUT_DIR=/tmp/higs_scalar_adj_freeze_results

echo "=== H2-BWD-2R Part A: SCALAR_ADJOINT Freeze Smoke ==="
echo "=== Build + Resource Check + Timing ==="

$PY /tmp/h2_bwd_2r_freeze_smoke.py \
    --out-dir "$OUT_DIR" \
    --source "$SOURCE" \
    --core-so "$CORE_SO" \
    --build \
    --build-dir /tmp/higs_scalar_adj_freeze_cache \
    --gpu 4 \
    --warmup 20 \
    --measure 100 \
    --reps 5 \
    --seed 4200

echo ""
echo "=== Results ==="
echo "--- resources.json ---"
cat "$OUT_DIR/resources.json"
echo ""
echo "--- analysis.json ---"
cat "$OUT_DIR/analysis.json"
echo ""
echo "--- production_timing.csv ---"
cat "$OUT_DIR/production_timing.csv"
