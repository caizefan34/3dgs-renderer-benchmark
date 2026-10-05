#!/bin/bash
set -u
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118
export TORCH_CUDA_ARCH_LIST=8.0
export BES0_PACKAGE_ROOT=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
export AABB_ALLOW_BUSY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6
export PATH=/home/liaoyuanjun/miniforge3/envs/nerficg/bin:/usr/bin:/bin:$PATH
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
DSH=/mnt/storage_pool/liaoyuanjun/bes0_mx/BES0_MX_DSH
cd /dev/shm/bes_aabb || exit 1
mkdir -p out/sweep
echo "[sweep6] START $(date -u +%H:%M:%S) inflate=$1"
export AABB_INFLATE=$1
export BES0_OUTPUT_DIR=out/sweep
"$PY" aabb_probe.py --scene garden --manifest "$DSH/config/scene_manifest.json" --gpu-index 6 2>&1 | sed "s/^/[infl$1] /"
echo "[sweep6] DONE inflate=$1 rc=$?"
