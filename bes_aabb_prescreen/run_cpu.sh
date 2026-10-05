#!/bin/bash
set -u
export BES0_PACKAGE_ROOT=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
export PATH=/home/liaoyuanjun/miniforge3/envs/nerficg/bin:/usr/bin:/bin:$PATH
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
DSH=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
cd /dev/shm/bes_aabb || exit 1
for SC in room bicycle garden; do
  echo "===START $SC $(date -u +%H:%M:%S) ==="
  "$PY" prescreen_aabb.py --scene "$SC" --device cpu --manifest "$DSH/config/scene_manifest.json"
  echo "===END $SC rc=$? $(date -u +%H:%M:%S) ==="
done
echo "===ALL DONE $(date -u +%H:%M:%S) ==="
