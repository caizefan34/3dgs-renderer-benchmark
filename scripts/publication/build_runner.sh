#!/bin/bash
set -x
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/usr/local/cuda/bin:$PATH
which ninja nvcc

echo "===== B1 build (pristine gsplat v1.5.3) ====="
cd /mnt/storage_pool/liaoyuanjun/pubphase
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python b1_clean_build.py 2>&1 | tail -8
echo "B1_RC=$?"

echo "===== B0 build: diff-gaussian-rasterization (original) ====="
cd /mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub
git rev-parse HEAD
git submodule status | head -5
cd submodules/diff-gaussian-rasterization
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python setup.py build_ext --inplace 2>&1 | tail -5
find . -name '*.so' | head -3
echo "B0_RASTER_RC=$?"

echo "===== B0 build: simple-knn ====="
cd ../simple-knn
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python setup.py build_ext --inplace 2>&1 | tail -4
find . -name '*.so' | head -3
echo "B0_KNN_RC=$?"
echo "ALL_BUILDS_DONE"
