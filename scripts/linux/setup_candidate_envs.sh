#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${MINIFORGE_HOME:-$HOME/miniforge3}/envs"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$HOME/3dgs-renderer-candidates}"
BASE_ENV="$ENV_ROOT/gsplat"

if [[ ! -x "$BASE_ENV/bin/python" ]]; then
  echo "missing base gsplat environment: $BASE_ENV" >&2
  exit 1
fi

mkdir -p "$CANDIDATE_ROOT"

install_candidate() {
  local name="$1" url="$2" commit="$3" subdir="$4" env_name="$5"
  local checkout="$CANDIDATE_ROOT/$name" env="$ENV_ROOT/$env_name"
  if [[ ! -d "$checkout/.git" ]]; then
    git init -q "$checkout"
    git -C "$checkout" remote add origin "$url"
  fi
  git -C "$checkout" fetch -q --depth 1 origin "$commit"
  git -C "$checkout" switch -q --detach FETCH_HEAD
  [[ "$(git -C "$checkout" rev-parse HEAD)" == "$commit" ]]
  if [[ ! -x "$env/bin/python" ]]; then
    "${MINIFORGE_HOME:-$HOME/miniforge3}/bin/conda" create -y -p "$env" --clone "$BASE_ENV"
  fi
  (
    cd "$checkout/$subdir"
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
    MAX_JOBS="${MAX_JOBS:-8}" \
      "$env/bin/python" -m pip install --no-build-isolation .
  )
}

install_candidate \
  flashgs https://github.com/InternLandMark/FlashGS \
  cdfc4e4002318423eda356eed02df8e01fa32cb6 . flashgs
install_candidate \
  local_gs https://github.com/tilaba/Local-GS \
  0c6d9e4a2cc458de90d3dc40753187d6d03ea514 \
  submodules/diff-gaussian-rasterization localgs
install_candidate \
  gemm_gs https://github.com/shieldforever/GEMM-GS \
  aca61f897f58964ff7204e1e3c6485995b5f212c submodules/gemm-gs gemmgs

for spec in \
  "flashgs flash_gaussian_splatting" \
  "localgs diff_gaussian_rasterization" \
  "gemmgs diff_gaussian_rasterization"; do
  read -r env_name module <<<"$spec"
  "$ENV_ROOT/$env_name/bin/python" -c "import torch; import $module; assert torch.cuda.is_available()"
done

echo "Candidate renderer environments are ready under $ENV_ROOT"
