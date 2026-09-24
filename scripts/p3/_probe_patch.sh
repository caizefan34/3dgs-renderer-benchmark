#!/bin/bash
# Create patched build.py (module name) + ext.cpp (torch namespace) for each probe.
set -e
FROZEN=/mnt/storage_pool/liaoyuanjun/higs_h8_mr_worktree/gsplat
DIAG=/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat
BASE=/mnt/storage_pool/liaoyuanjun/p3h_probe

function patch_probe() {
  local tag=$1 src=$2 module_name=$3
  local cud="${src}/experimental/render/kernels/cuda"
  local dst="${BASE}/${tag}/experimental/render/kernels/cuda"
  # build.py: copy and rename module name
  cp "$cud/build.py" "$dst/build.py"
  sed -i "s/experimental_gaussian_render_inference_scene_cuda/${module_name}/g" "$dst/build.py"
  # ext.cpp: copy and rename torch namespace
  cp "$cud/ext.cpp" "$dst/ext.cpp"
  sed -i "s/TORCH_LIBRARY(experimental, m)/TORCH_LIBRARY(p3h_${tag}, m)/" "$dst/ext.cpp"
  sed -i "s/TORCH_LIBRARY_IMPL(experimental, CUDA, m)/TORCH_LIBRARY_IMPL(p3h_${tag}, CUDA, m)/" "$dst/ext.cpp"
  sed -i "s/TORCH_LIBRARY_IMPL(experimental, Autograd, m)/TORCH_LIBRARY_IMPL(p3h_${tag}, Autograd, m)/" "$dst/ext.cpp"
  echo "patched $tag (module=$module_name)"
}

# prod: python module name "experimental_p3h_prod_cuda", torch ns p3h_prod
patch_probe probe_prod "$FROZEN" "experimental_p3h_prod_cuda"
# v0:   python module name "experimental_p3h_v0_cuda",   torch ns p3h_v0
patch_probe probe_v0   "$DIAG"   "experimental_p3h_v0_cuda"

grep -n 'experimental_p3h_prod\|experimental_p3h_v0' "$BASE"/*/experimental/render/kernels/cuda/build.py
echo "---- ext.cpp namespaces ----"
grep -n 'TORCH_LIBRARY' "$BASE"/*/experimental/render/kernels/cuda/ext.cpp