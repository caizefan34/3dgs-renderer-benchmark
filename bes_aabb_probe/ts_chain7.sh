#!/bin/bash
set -u
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118
export TORCH_CUDA_ARCH_LIST=8.0
export BES0_PACKAGE_ROOT=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
export AABB_ALLOW_BUSY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7
export PATH=/home/liaoyuanjun/miniforge3/envs/nerficg/bin:/usr/bin:/bin:$PATH
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
DSH=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
cd /dev/shm/bes_aabb || exit 1
echo "[ts7] START $(date -u +%H:%M:%S) CUDA_VISIBLE_DEVICES=7"
for SC in room bicycle garden; do
  echo "=== TS scene=$SC start $(date -u +%H:%M:%S) ==="
  "$PY" aabb2_probe.py --scene "$SC" --manifest "$DSH/config/scene_manifest.json" --gpu-index 7
  echo "=== TS scene=$SC rc=$? end $(date -u +%H:%M:%S) ==="
done
echo "[ts7] DONE $(date -u +%H:%M:%S)"
touch /dev/shm/bes_aabb/TS7_DONE
