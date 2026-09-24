#!/bin/bash
# B1_CLEAN_GSPLAT build: pristine upstream gsplat v1.5.3 (937e2991) for the
# publication P1 baseline. Method: copy the accutile tree (which is upstream
# v1.5.3 at HEAD + a 9-file uncommitted accutile patch), git-revert the 9
# patched files -> git-verified pristine v1.5.3 with glm submodule present.
# Then build via the b1a_build.py pattern with a distinct extension name.
set -x
ACC=/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153
B1TREE=/mnt/storage_pool/liaoyuanjun/gsplat-b1-clean-v153
CACHE=/mnt/storage_pool/liaoyuanjun/gsplat_b1clean_pub_cache
NAME=gsplat_cuda_b1clean

echo "== disk =="
df -h /mnt/storage_pool | tail -1

if [ ! -d "$B1TREE" ]; then
  cp -a "$ACC" "$B1TREE"
fi
cd "$B1TREE" || exit 1
echo "== revert the 9 accutile-patched files to HEAD (pristine v1.5.3) =="
git checkout -- gsplat/cuda/_backend.py gsplat/cuda/_wrapper.py \
  gsplat/cuda/csrc/Intersect.cpp gsplat/cuda/csrc/Intersect.h \
  gsplat/cuda/csrc/IntersectTile.cu gsplat/cuda/include/Common.h \
  gsplat/cuda/include/Ops.h gsplat/rendering.py setup.py 2>&1
git status --ignore-submodules --short | head -5
echo "== pristine checks =="
echo "accutile hits: $(grep -rn 'accutile' gsplat/rendering.py gsplat/cuda/_wrapper.py gsplat/cuda/csrc/IntersectTile.cu | wc -l)  (must be 0)"
grep -n 'absgrad: bool = False' gsplat/rendering.py | head -2
git rev-parse HEAD
git describe --tags 2>/dev/null

echo "== build =="
export PATH=/usr/local/cuda/bin:$PATH
which nvcc
cd /mnt/storage_pool/liaoyuanjun/pubphase
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python b1_clean_build.py 2>&1 | tail -25
echo "B1_BUILD_DONE rc=$?"
