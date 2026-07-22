#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINIFORGE_HOME="${MINIFORGE_HOME:-$HOME/miniforge3}"
ENV_ROOT="$MINIFORGE_HOME/envs"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$HOME/renderer-candidates}"
BASE_ENV="$ENV_ROOT/gsplat"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

while [[ ! -x "$BASE_ENV/bin/python" ]]; do
  echo "waiting for base gsplat environment: $BASE_ENV"
  sleep 30
done

MINIFORGE_HOME="$MINIFORGE_HOME" CANDIDATE_ROOT="$CANDIDATE_ROOT" \
  CUDA_HOME="$CUDA_HOME" bash "$ROOT/scripts/linux/setup_candidate_envs.sh"

original="$CANDIDATE_ROOT/original_3dgs_train"
if [[ ! -d "$original/.git" ]]; then
  git init -q "$original"
  git -C "$original" remote add origin https://github.com/graphdeco-inria/gaussian-splatting
fi
git -C "$original" fetch -q --depth 1 origin 54c035f7834b564019656c3e3fcc3646292f727d
git -C "$original" switch -q --detach FETCH_HEAD
git -C "$original" submodule update --init --depth 1 \
  submodules/simple-knn submodules/diff-gaussian-rasterization submodules/fused-ssim

git -C "$CANDIDATE_ROOT/local_gs" submodule update --init --depth 1 submodules/simple-knn
git -C "$CANDIDATE_ROOT/gemm_gs" submodule update --init --depth 1 \
  submodules/simple-knn submodules/fused-ssim

clone_env() {
  local source="$1" destination="$2"
  if [[ ! -x "$ENV_ROOT/$destination/bin/python" ]]; then
    "$MINIFORGE_HOME/bin/conda" create -y -p "$ENV_ROOT/$destination" \
      --clone "$ENV_ROOT/$source"
  fi
  "$ENV_ROOT/$destination/bin/python" -m pip install \
    plyfile tqdm opencv-python-headless joblib
}

clone_env gsplat train_original
clone_env localgs train_localgs
clone_env gemmgs train_gemmgs

export CUDA_HOME TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
  MAX_JOBS="${MAX_JOBS:-8}"
"$ENV_ROOT/train_original/bin/python" -m pip install --no-build-isolation \
  "$original/submodules/diff-gaussian-rasterization" \
  "$original/submodules/simple-knn" "$original/submodules/fused-ssim"
"$ENV_ROOT/train_localgs/bin/python" -m pip install --no-build-isolation \
  "$CANDIDATE_ROOT/local_gs/submodules/simple-knn"
"$ENV_ROOT/train_gemmgs/bin/python" -m pip install --no-build-isolation \
  "$CANDIDATE_ROOT/gemm_gs/submodules/simple-knn" \
  "$CANDIDATE_ROOT/gemm_gs/submodules/fused-ssim"

for specification in \
  "train_original diff_gaussian_rasterization simple_knn" \
  "train_localgs diff_gaussian_rasterization simple_knn" \
  "train_gemmgs diff_gaussian_rasterization simple_knn"; do
  read -r environment rasterizer knn <<<"$specification"
  "$ENV_ROOT/$environment/bin/python" -c \
    "import torch; import $rasterizer; import $knn; assert torch.cuda.is_available()"
done

echo "Native training environments are ready under $ENV_ROOT"
