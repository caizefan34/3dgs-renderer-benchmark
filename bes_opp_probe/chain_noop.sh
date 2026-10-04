#!/bin/bash
set -u
GPUIDX="${1:-7}"
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
DSH=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
OUT=/dev/shm/bes_opp/out
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118
export BES0_PACKAGE_ROOT="$DSH"
export BES0_OUTPUT_DIR="$OUT"
export PATH="$CUDA_HOME/bin:$(dirname "$PY"):$PATH"
cd /dev/shm/bes_opp || exit 1
for SC in room bicycle garden; do
  echo "=== NOOP scene=$SC start $(date -u +%H:%M:%S) ==="
  "$PY" noop_census.py --scene "$SC" --manifest "$DSH/config/scene_manifest.json" --gpu-index "$GPUIDX"
  echo "=== NOOP scene=$SC end rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== NOOP ALL DONE $(date -u +%H:%M:%S) ==="
