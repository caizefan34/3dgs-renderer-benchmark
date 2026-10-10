#!/usr/bin/env bash
# R19 environment identity + preflight capture.
set -uo pipefail
HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export HIGS_PROJECT="${HIGS_PROJECT:-/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx}"
export R19_OUT="${R19_OUT:-/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_FULLTRAIN_R19_MX}"
export PYTHON="${PYTHON:-/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3}"
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-/mnt/storage_pool/liaoyuanjun/tmp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
mkdir -p "$R19_OUT"/{logs,raw,scripts,captures}
PY="$PYTHON"
{
  echo "== R19 environment identity =="
  date -u
  echo "host: $(hostname)"
  echo "python: $($PY -c 'import sys;print(sys.executable, sys.version.split()[0])' 2>&1)"
  echo "torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>&1)"
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
  df -h /mnt/storage_pool 2>&1 | tail -1
} | tee "$R19_OUT/logs/runner.log"

$PY "$HERE/r19_preflight.py" 2>&1 | tee "$R19_OUT/logs/preflight.log"
echo "PREFLIGHT_EXIT=${PIPESTATUS[0]}"
