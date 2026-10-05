#!/bin/bash
set -u
GPUIDX="${1:-7}"
export CUDA_VISIBLE_DEVICES="$GPUIDX"
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
DSH=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
OUT=/dev/shm/bes_aabb/out
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118
export TORCH_CUDA_ARCH_LIST="8.0"
export BES0_PACKAGE_ROOT="$DSH"
export BES0_OUTPUT_DIR="$OUT"
export AABB_REPEATS="${AABB_REPEATS:-50}"
export PATH=/home/liaoyuanjun/miniforge3/envs/nerficg/bin:/usr/bin:/bin:$PATH
mkdir -p "$OUT"
cd /dev/shm/bes_aabb || exit 1
for SC in room bicycle garden; do
  echo "=== AABB scene=$SC start $(date -u +%H:%M:%S) ==="
  "$PY" aabb_probe.py --scene "$SC" --manifest "$DSH/config/scene_manifest.json" --gpu-index "$GPUIDX"
  echo "=== AABB scene=$SC end rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== AABB ALL DONE $(date -u +%H:%M:%S) ==="
