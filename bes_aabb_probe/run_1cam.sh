#!/bin/bash
set -u
GPUIDX="${1:-7}"
export CUDA_VISIBLE_DEVICES="$GPUIDX"
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
DSH=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118
export TORCH_CUDA_ARCH_LIST="8.0"
export BES0_PACKAGE_ROOT="$DSH"
export BES0_OUTPUT_DIR=/dev/shm/bes_aabb/out
export AABB_REPEATS=50
export PATH=/home/liaoyuanjun/miniforge3/envs/nerficg/bin:/usr/bin:/bin:$PATH
cd /dev/shm/bes_aabb || exit 1
"$PY" aabb_probe.py --scene room --manifest "$DSH/config/scene_manifest.json" --gpu-index "$GPUIDX" --max-cams 2
