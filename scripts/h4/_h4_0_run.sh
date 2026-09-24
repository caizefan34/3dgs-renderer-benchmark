#!/bin/bash
# H4-0 Run script (mx host, frozen H1/H2 authoritative environment = higs-13scene-env)
set -e
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/home/liaoyuanjun/.local/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=3

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
AL=/tmp/h3_fwd_0_al.py
OUT_DIR=/tmp/higs_h4_0

echo "=== H4-0: Exact Pixel-Support Spatial Sparsity Oracle ==="
echo "  PY: $PY   AL: $AL   OUT: $OUT_DIR   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

mkdir -p "$OUT_DIR"

$PY /tmp/h4_0_pixel_sparsity_oracle.py \
    --out-dir "$OUT_DIR" \
    --al "$AL" \
    --scenes garden \
    --gpu 3 \
    --max-lo-side 2048 \
    --seed 4200

echo ""
echo "=== Results ==="
cat "$OUT_DIR/pixel_sparsity_summary.json"
