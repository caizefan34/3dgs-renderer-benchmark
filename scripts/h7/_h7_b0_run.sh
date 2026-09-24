#!/bin/bash
set -e
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=4
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
echo "ninja: $(ninja --version 2>&1 | head -1)"
echo "CUDA_DEVICES=4"
$PY /tmp/h7_b0_roofline_oracle.py --out-dir /tmp/higs_h7_b0 --gpu 4 --max-long-side 2048 2>&1
echo "=== EXIT $? ==="