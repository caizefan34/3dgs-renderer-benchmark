#!/usr/bin/env bash
set -euo pipefail
MINIFORGE_HOME="${MINIFORGE_HOME:-$HOME/miniforge3}"
ENV_ROOT="$MINIFORGE_HOME/envs"
CANDIDATE="${CANDIDATE:-/root/renderer-candidates/original_3dgs_train}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" MAX_JOBS="${MAX_JOBS:-8}"

if [[ ! -x "$ENV_ROOT/train_original/bin/python" ]]; then
  "$MINIFORGE_HOME/bin/conda" create -y -p "$ENV_ROOT/train_original" --clone "$ENV_ROOT/gsplat"
fi
"$ENV_ROOT/train_original/bin/python" -m pip install -q plyfile tqdm opencv-python-headless joblib
"$ENV_ROOT/train_original/bin/python" -m pip install --no-build-isolation -q \
  "$CANDIDATE/submodules/diff-gaussian-rasterization" \
  "$CANDIDATE/submodules/simple-knn" \
  "$CANDIDATE/submodules/fused-ssim"
"$ENV_ROOT/train_original/bin/python" -c \
  "import torch, diff_gaussian_rasterization, simple_knn, fused_ssim; assert torch.cuda.is_available()"
echo TRAIN_ORIGINAL_ENV_READY
