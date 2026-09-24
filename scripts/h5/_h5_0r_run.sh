#!/bin/bash
set -e
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/home/liaoyuanjun/.local/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=4

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
OUT_DIR=/tmp/higs_h5_0r

echo "=== H5-0R: Frontier-Aware Exact Backward Load Gating ==="

$PY /tmp/h5_0r_frontier_load_gating.py \
    --out-dir "$OUT_DIR" \
    --gpu 4 \
    --max-long-side 2048

echo ""
echo "=== Done ==="
