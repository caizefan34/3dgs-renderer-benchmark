#!/bin/bash
# H2-BWD-1R: Oracle Integrity Repair launcher for mx
set -e

export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=3

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
LOCAL_DIR=/mnt/storage_pool/3dgs-renderer-benchmark
OUT_DIR=$LOCAL_DIR/artifacts/higs-h2-bwd-1r

mkdir -p $OUT_DIR

echo "===== H2-BWD-1R: Oracle Integrity Repair (room/cam0) ====="
$PY $LOCAL_DIR/scripts/h2/h2_bwd_1r_oracle_integrity.py \
    --out-dir $OUT_DIR \
    --scene room --cam-idx 0 --gpu 0 --max-long-side 2048 \
    --n-warmup 20 --n-measure 100 --n-reps 5 --seed 42 \
    2>&1 | tee $OUT_DIR/room_cam0.log

echo ""
echo "===== H2-BWD-1R DONE ====="
