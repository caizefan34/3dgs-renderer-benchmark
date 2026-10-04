#!/bin/bash
set -u
SCENE="$1"; GPUIDX="${2:-6}"
shift 2 2>/dev/null || true
EXTRA_ARGS=("$@")
W=/mnt/storage_pool/liaoyuanjun/bes0_mx
DSH="$W/BES0_MX_DSH"
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
OUT=/dev/shm/bes_opp/out
mkdir -p "$OUT"
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader -i "$GPUIDX" | awk -F', ' '{print $2}')
echo "[run_opp] index=$GPUIDX uuid=$UUID scene=$SCENE extra=${EXTRA_ARGS[*]:-}"
export CUDA_VISIBLE_DEVICES="$UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118
export BES0_PACKAGE_ROOT="$DSH"
export BES0_OUTPUT_DIR="$OUT"
export PATH="$CUDA_HOME/bin:$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
cd /dev/shm/bes_opp
ARGS=(--scene "$SCENE" --manifest "$DSH/config/scene_manifest.json" --gpu-index "$GPUIDX")
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then ARGS+=("${EXTRA_ARGS[@]}"); fi
exec "$PY" probe_opportunity.py "${ARGS[@]}"
