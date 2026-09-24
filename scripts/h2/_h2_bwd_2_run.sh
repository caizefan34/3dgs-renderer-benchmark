#!/bin/bash
# H2-BWD-2: SCALAR_ADJOINT Production Validation launcher for mx
# Reuses the exact frozen CF binary via TORCH_EXTENSIONS_DIR (never rebuilds).
set -e

export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
# Reuse the EXACT frozen production binary (sha256 da5300...), never rebuild.
export TORCH_EXTENSIONS_DIR=/tmp/higs_h2_bwd_cf/cache
export HIGS_PX_RUNTIME=2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
LOCAL_DIR=/mnt/storage_pool/3dgs-renderer-benchmark
WORK=/tmp/higs_h2_bwd_2
OUT_DIR=$WORK/artifacts
mkdir -p $WORK $OUT_DIR

SCENES=${SCENES:-room,bicycle,garden}
WARMUP=${WARMUP:-20}
MEASURE=${MEASURE:-100}
REPS=${REPS:-5}
SEED=${SEED:-4200}
GPU=${GPU:-0}
NCU_BIN=${NCU_BIN:-/usr/bin/ncu}

echo "===== H2-BWD-2: SCALAR_ADJOINT Production Validation ====="
echo "  scenes=$SCENES gpu=$GPU warmup=$WARMUP measure=$MEASURE reps=$REPS seed=$SEED"
echo "  binary cache: $TORCH_EXTENSIONS_DIR"

$PY $WORK/h2_bwd_2_validation.py \
    --out-dir $OUT_DIR \
    --scenes "$SCENES" \
    --cam-idx 0 \
    --gpu $GPU \
    --max-long-side 2048 \
    --warmup $WARMUP \
    --measure $MEASURE \
    --reps $REPS \
    --seed $SEED \
    --source /tmp/higs_h2_bwd_cf/source \
    --core-so /tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so \
    --ptxas-log /tmp/higs_h2_bwd_cf/build.log \
    --binary /tmp/higs_h2_bwd_cf/cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so \
    --ncu "$NCU_BIN" \
    --fb-scenes "room,bicycle" \
    2>&1 | tee $OUT_DIR/run.log

echo ""
echo "===== H2-BWD-2 DONE ====="
echo "Artifacts in: $OUT_DIR"
