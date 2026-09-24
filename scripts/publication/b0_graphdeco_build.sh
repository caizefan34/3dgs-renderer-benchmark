#!/bin/bash
# B0 Graphdeco build: clone official graphdeco-inria/gaussian-splatting at the
# frozen reference commit 54c035f (the commit reference_v1's optimizer values
# are documented against), recursive submodules, then build the original
# diff-gaussian-rasterization CUDA package under higs-13scene-env.
set -x
export PATH=/usr/local/cuda/bin:$PATH
B0TREE=/mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub
COMMIT=54c035f

echo "== disk =="
df -h /mnt/storage_pool | tail -1

if [ ! -d "$B0TREE" ]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting "$B0TREE" 2>&1 | tail -5
fi
cd "$B0TREE" || exit 1
git fetch --tags 2>/dev/null
git checkout "$COMMIT" 2>&1 | tail -2
git submodule update --init --recursive 2>&1 | tail -3
echo "== identity =="
git rev-parse HEAD
git log -1 --format="%H %ad %s" --date=short
git submodule status

echo "== build diff-gaussian-rasterization (original) =="
cd "$B0TREE/submodules/diff-gaussian-rasterization" || exit 1
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python setup.py build_ext --inplace 2>&1 | tail -6
ls build/*/gsplat_cuda*.so 2>/dev/null
ls *.so 2>/dev/null
echo "B0_RASTERIZER_BUILD rc=$?"

echo "== build simple-knn (scale init dependency check later) =="
cd "$B0TREE/submodules/simple-knn" || exit 1
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python setup.py build_ext --inplace 2>&1 | tail -4
ls *.so 2>/dev/null
echo "B0_SIMPLE_KNN rc=$?"
echo "B0_BUILD_DONE"
