#!/bin/bash
set -u
GPUIDX="${1:-7}"
W=/dev/shm/bes_opp
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
DSH=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
OUT=$W/out; mkdir -p "$OUT"
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118
export BES0_PACKAGE_ROOT="$DSH"
export BES0_OUTPUT_DIR="$OUT"
export PATH="$CUDA_HOME/bin:$(dirname $PY):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
cd "$W" || exit 1
for SC in room bicycle garden; do
  echo "=== HEADROOM scene=$SC start $(date -u +%H:%M:%S) ==="
  "$PY" headroom_sweep.py --scene "$SC" --manifest "$DSH/config/scene_manifest.json" --gpu-index "$GPUIDX"
  echo "=== HEADROOM scene=$SC end rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== HEADROOM ALL DONE $(date -u +%H:%M:%S) ==="
