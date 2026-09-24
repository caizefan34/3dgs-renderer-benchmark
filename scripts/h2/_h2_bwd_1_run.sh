#!/bin/bash
# H2-BWD-1: Compute-Factored HiGS Backward Oracle launcher for mx
set -e

export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=3

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
LOCAL_DIR=/mnt/storage_pool/3dgs-renderer-benchmark
OUT_DIR=$LOCAL_DIR/artifacts/higs-h2-bwd-1

mkdir -p $OUT_DIR

echo "===== H2-BWD-1: Compute Oracle (room/cam0) ====="
$PY $LOCAL_DIR/scripts/h2/h2_bwd_1_compute_oracle.py \
    --out $OUT_DIR/room_cam0_compute_oracle.json \
    --scene room --cam-idx 0 --gpu 0 --max-long-side 2048 \
    --n-warmup 20 --n-measure 100 \
    2>&1 | tee $OUT_DIR/room_cam0.log

echo ""
echo "===== H2-BWD-1: Compute Oracle (bicycle/cam0) ====="
$PY $LOCAL_DIR/scripts/h2/h2_bwd_1_compute_oracle.py \
    --out $OUT_DIR/bicycle_cam0_compute_oracle.json \
    --scene bicycle --cam-idx 0 --gpu 0 --max-long-side 2048 \
    --n-warmup 20 --n-measure 100 \
    2>&1 | tee $OUT_DIR/bicycle_cam0.log || echo "bicycle skipped"

echo ""
echo "===== H2-BWD-1 DONE ====="
