#!/bin/bash
# Build two renamed-namespace probe .so that can coexist in ONE torch process:
#   probe_prod : frozen C0 V3 kernel  (source = higs_h8_mr_worktree)   -> name experimental_p3h_prod_cuda, ns p3h_prod
#   probe_v0   : diag V0 P3=0 kernel  (source = higs_p3h_worktree)     -> name experimental_p3h_v0_cuda,   ns p3h_v0
# Kernel source dirs are symlinked; only ext.cpp (namespace) and build.py (name) differ.
set -e
ENV=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
FROZEN=/mnt/storage_pool/liaoyuanjun/higs_h8_mr_worktree/gsplat
DIAG=/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat
BASE=/mnt/storage_pool/liaoyuanjun/p3h_probe
rm -rf "$BASE"
mkdir -p "$BASE"

function make_probe() {
  local tag=$1 src=$2 mkdirs
  local root="$BASE/$tag"
  # layout: <root>/experimental/render/kernels/cuda/{build.py,ext.cpp,csrc}
  #         <root>/cuda  (symlink for include  +  third_party/glm  +  csrc/Config.h)
  mkdir -p "$root/experimental/render/kernels/cuda"
  ln -s "$src/cuda" "$root/cuda"
  ln -s "$src/experimental/render/kernels/cuda/csrc" "$root/experimental/render/kernels/cuda/csrc"
}

# ---- sources (symlink whole csrc/kernel tree) ------------------------------
make_probe probe_prod "$FROZEN"
make_probe probe_v0   "$DIAG"

echo "probe dirs built."
ls -l "$BASE/probe_prod/" "$BASE/probe_prod/experimental/render/kernels/cuda/"