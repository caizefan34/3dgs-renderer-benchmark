#!/bin/bash
set -e
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/home/liaoyuanjun/.local/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=4

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
SOURCE=/tmp/higs_h3_fwd_1a_source
CORE_SO=/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so
OUT_DIR=/tmp/higs_h5_0

echo "=== H5-0: Last-ID Guided Hierarchical Backward Pruning Oracle ==="

$PY /tmp/h5_0_lastid_pruning_oracle.py \
    --out-dir "$OUT_DIR" \
    --source "$SOURCE" \
    --core-so "$CORE_SO" \
    --gpu 4 \
    --max-long-side 2048

echo ""
echo "=== Done ==="
echo "Decision:"
python3 -c "import json; d=json.load(open('$OUT_DIR/analysis.json')); print(d['gate'])" 2>/dev/null
