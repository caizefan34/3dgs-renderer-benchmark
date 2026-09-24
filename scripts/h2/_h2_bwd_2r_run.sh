#!/bin/bash
set -e
export PATH=/home/liaoyuanjun/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=4
export HIGS_PX_RUNTIME=2
export TORCH_EXTENSIONS_DIR=/tmp/higs_h2_bwd_cf/cache
export HIGS_BWD_CF_VARIANT=baseline

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
OUT=/tmp/higs_h2_bwd_2r/artifacts

mkdir -p $OUT

echo "=== H2-BWD-2R launcher ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $PY"
echo "Output: $OUT"

$PY /tmp/higs_h2_bwd_2r/h2_bwd_2r_exactness.py \
  --out-dir $OUT \
  --scenes room,bicycle,garden \
  --cam-idx 0 \
  --gpu 4 \
  --max-long-side 2048 \
  --seed 4200 \
  --source /tmp/higs_h2_bwd_cf/source \
  --core-so /tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so \
  --exp-so /tmp/higs_h2_bwd_cf/instrumented-cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so \
  --noise-runs 10 \
  --max-oracle-tiles 20 \
  --max-vjp-samples 200 \
  2>&1 | tee $OUT/run.log
