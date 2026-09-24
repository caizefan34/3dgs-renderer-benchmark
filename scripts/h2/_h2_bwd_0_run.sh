#!/bin/bash
# H2-BWD-0: Structural analysis + atomic-free oracle on mx
set -e

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=3

LOCAL_DIR=/mnt/storage_pool/3dgs-renderer-benchmark
OUT_DIR=$LOCAL_DIR/artifacts/higs-h2-bwd-0

mkdir -p $OUT_DIR

echo "===== ENV CHECK ====="
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
$PY -c "from gsplat.experimental import rasterize_gaussian_higs_frozen; print('B2 import OK')"
$PY -c "from gsplat.rendering import _maybe_evaluate_sh; print('SH import OK')"

echo ""
echo "===== H2-BWD-0 Step 1: Structural analysis (room/cam0) ====="
$PY $LOCAL_DIR/scripts/h2/h2_bwd_0_structural.py \
    --out-dir $OUT_DIR \
    --scene room \
    --cam-idx 0 \
    --gpu 0 \
    --max-long-side 2048 \
    2>&1 | tee $OUT_DIR/structural_room.log

echo ""
echo "===== H2-BWD-0 Step 2: Atomic-free oracle (room/cam0) ====="
$PY $LOCAL_DIR/scripts/h2/h2_bwd_0_atomic_oracle.py \
    --out $OUT_DIR/room_cam0_atomic_oracle.json \
    --scene room \
    --cam-idx 0 \
    --gpu 0 \
    --max-long-side 2048 \
    --n-warmup 20 \
    --n-measure 100 \
    2>&1 | tee $OUT_DIR/atomic_oracle_room.log

echo ""
echo "===== H2-BWD-0 Step 3: Optional bicycle/cam0 structural stats ====="
$PY $LOCAL_DIR/scripts/h2/h2_bwd_0_structural.py \
    --out-dir $OUT_DIR \
    --scene bicycle \
    --cam-idx 0 \
    --gpu 0 \
    --max-long-side 2048 \
    2>&1 | tee $OUT_DIR/structural_bicycle.log || echo "bicycle skipped (optional)"

echo ""
echo "===== H2-BWD-0 DONE ====="
