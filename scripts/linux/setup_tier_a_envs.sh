#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINIFORGE_HOME="${MINIFORGE_HOME:-$HOME/miniforge3}"
ENV_ROOT="$MINIFORGE_HOME/envs"
SOURCE_ROOT="$ROOT/artifacts/renderer-sources"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MINIFORGE_VERSION="${MINIFORGE_VERSION:-25.3.1-0}"
PYTORCH_VERSION="2.9.1+cu128"
TORCHVISION_VERSION="0.24.1+cu128"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYTHONNOUSERSITE=1

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup must run on native Linux." >&2
  exit 1
fi
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "Ubuntu 22.04 LTS is required (found ${PRETTY_NAME:-unknown})." >&2
  exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
  APT=(apt-get)
elif command -v sudo >/dev/null; then
  APT=(sudo apt-get)
else
  echo "Run as root or install sudo before setting up system packages." >&2
  exit 1
fi
"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends \
  build-essential ca-certificates cmake curl git ninja-build pkg-config rsync tmux

command -v nvidia-smi >/dev/null || {
  echo "Install the Ubuntu-recommended proprietary NVIDIA driver first." >&2
  exit 1
}
[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
  echo "A CUDA Toolkit with nvcc is required at $CUDA_HOME." >&2
  exit 1
}

if [[ ! -x "$MINIFORGE_HOME/bin/conda" ]]; then
  installer="$(mktemp --suffix=.sh)"
  curl -fL --retry 5 \
    "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    -o "$installer"
  bash "$installer" -b -p "$MINIFORGE_HOME"
  rm -f "$installer"
fi
source "$MINIFORGE_HOME/etc/profile.d/conda.sh"

create_env() {
  local name="$1"
  if [[ ! -x "$ENV_ROOT/$name/bin/python" ]]; then
    conda create -y -p "$ENV_ROOT/$name" python=3.10 pip
  else
    conda install -y -p "$ENV_ROOT/$name" python=3.10 pip
  fi
  "$ENV_ROOT/$name/bin/python" -c \
    "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
  "$ENV_ROOT/$name/bin/python" -m pip install --upgrade pip setuptools wheel ninja
  "$ENV_ROOT/$name/bin/python" -m pip install \
    "torch==$PYTORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url https://download.pytorch.org/whl/cu128
  "$ENV_ROOT/$name/bin/python" -m pip install \
    -r "$ROOT/requirements-benchmark.txt" \
    -r "$ROOT/requirements-quality.txt" \
    pytest==9.1.1
  "$ENV_ROOT/$name/bin/python" -m pip install -e "$ROOT"
}

for env_name in original3dgs gsplat speedy tcgs; do
  create_env "$env_name"
done

prepare_source() {
  local directory="$1"
  local url="$2"
  local commit="$3"
  shift 3
  if [[ -d "$directory/.git" ]]; then
    [[ -z "$(git -C "$directory" status --porcelain --untracked-files=no --ignore-submodules=untracked)" ]] || {
      echo "Renderer checkout has modified tracked files: $directory" >&2
      exit 1
    }
    git -C "$directory" fetch --all --tags
  else
    git clone "$url" "$directory"
  fi
  git -C "$directory" checkout --detach "$commit"
  if [[ "$#" -gt 0 ]]; then
    git -C "$directory" submodule update --init --recursive -- "$@"
  fi
  [[ "$(git -C "$directory" rev-parse HEAD)" == "$commit" ]]
}

mkdir -p "$SOURCE_ROOT"
prepare_source \
  "$SOURCE_ROOT/original-diff-gaussian-rasterization" \
  https://github.com/graphdeco-inria/diff-gaussian-rasterization \
  9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0 \
  third_party/glm
prepare_source \
  "$SOURCE_ROOT/gsplat" \
  https://github.com/nerfstudio-project/gsplat \
  77ab983ffe43420b2131669cb35776b883ca4c3c \
  gsplat/cuda/csrc/third_party/glm
prepare_source \
  "$SOURCE_ROOT/speedy-splat" \
  https://github.com/j-alex-hanson/speedy-splat \
  34c45c6d9b8bd6110231864f2f358b6d3abbf73d \
  submodules/diff-gaussian-rasterization
prepare_source \
  "$SOURCE_ROOT/3DGSTensorCore" \
  https://github.com/DeepLink-org/3DGSTensorCore \
  0bb82f88fde211c34b42e1497f0fc7265461592b

"$ENV_ROOT/original3dgs/bin/python" -m pip install --no-build-isolation -e \
  "$SOURCE_ROOT/original-diff-gaussian-rasterization"
"$ENV_ROOT/gsplat/bin/python" -m pip install --no-build-isolation -e \
  "$SOURCE_ROOT/gsplat"

speedy_source="$SOURCE_ROOT/speedy-splat/submodules/diff-gaussian-rasterization"
(
  cd "$speedy_source"
  "$ENV_ROOT/speedy/bin/python" setup.py build_ext --inplace
)
speedy_alias="$SOURCE_ROOT/speedy-splat/speedy_gaussian_rasterization"
case "$speedy_alias" in
  "$SOURCE_ROOT/speedy-splat/"*) ;;
  *) echo "Unsafe Speedy alias path" >&2; exit 1 ;;
esac
rm -rf -- "$speedy_alias"
cp -a "$speedy_source/diff_gaussian_rasterization" "$speedy_alias"
echo "$SOURCE_ROOT/speedy-splat" > \
  "$ENV_ROOT/speedy/lib/python3.10/site-packages/speedy-splat-source.pth"

"$ENV_ROOT/tcgs/bin/python" -m pip install --no-build-isolation -e \
  "$SOURCE_ROOT/3DGSTensorCore/submodules/tcgs_speedy_rasterizer"

for env_name in original3dgs gsplat speedy tcgs; do
  "$ENV_ROOT/$env_name/bin/python" -m pip check
  "$ENV_ROOT/$env_name/bin/python" -c \
    "import torch; assert torch.__version__ == '$PYTORCH_VERSION'; assert torch.version.cuda == '12.8'; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability() == (8, 0); x=torch.ones(1024, device='cuda'); assert (x + x).sum().item() == 2048"
  "$ENV_ROOT/$env_name/bin/python" -c \
    "import lpips; lpips.LPIPS(net='alex')"
done

smoke_renderer() {
  local env_name="$1"
  local renderer="$2"
  local commit="$3"
  "$ENV_ROOT/$env_name/bin/python" -c \
    "import sys; sys.path.insert(0, '$ROOT/src'); from renderers import get_renderer_class; r=get_renderer_class('$renderer')(device='cuda'); assert r.is_available(); m=r.metadata(); assert m['commit_hash'] == '$commit', m; print('$renderer', m)"
}

smoke_renderer original3dgs original_3dgs 9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0
smoke_renderer gsplat gsplat 77ab983ffe43420b2131669cb35776b883ca4c3c
smoke_renderer gsplat gsplat_higs 77ab983ffe43420b2131669cb35776b883ca4c3c
smoke_renderer speedy speedy_splat 34c45c6d9b8bd6110231864f2f358b6d3abbf73d
smoke_renderer tcgs tcgs 0bb82f88fde211c34b42e1497f0fc7265461592b

garden_scene="$ROOT/datasets/processed/mipnerf360/garden/point_cloud.ply"
garden_cameras="$ROOT/datasets/processed/mipnerf360/garden/eval_cameras.json"
if [[ "${SMOKE_GARDEN_FRAME:-1}" == "1" && -f "$garden_scene" && -f "$garden_cameras" ]]; then
  smoke_garden_frame() {
    local env_name="$1"
    local renderer="$2"
    PYTHONPATH="$ROOT/src" "$ENV_ROOT/$env_name/bin/python" - \
      "$renderer" "$garden_scene" "$garden_cameras" <<'PY'
import sys
import torch
from benchmark_framework import load_cameras_from_json, load_ply
from renderers import get_renderer_class

renderer_id, scene_path, cameras_path = sys.argv[1:]
renderer = get_renderer_class(renderer_id)(device="cuda")
scene = renderer.prepare_scene(load_ply(scene_path, device="cuda"))
camera = load_cameras_from_json(cameras_path, device="cuda")[0]
with torch.inference_mode():
    image = renderer.render(scene, camera)
assert image.dtype == torch.float32, image.dtype
assert image.ndim == 3 and image.shape[-1] == 3, image.shape
assert image.is_cuda
print(renderer_id, tuple(image.shape), image.dtype, image.device)
PY
  }
  smoke_garden_frame original3dgs original_3dgs
  smoke_garden_frame gsplat gsplat
  smoke_garden_frame gsplat gsplat_higs
  smoke_garden_frame speedy speedy_splat
  smoke_garden_frame tcgs tcgs
else
  echo "Skipping garden frame smoke (canonical files absent or SMOKE_GARDEN_FRAME=0)."
fi

echo "Tier A renderer environments are ready under $ENV_ROOT"
echo "Dry run: $ENV_ROOT/gsplat/bin/python $ROOT/src/scripts/run_linux_tier_a_matrix.py --dry-run"
touch "$ENV_ROOT/.tier-a-ready"
