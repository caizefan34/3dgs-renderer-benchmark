#!/bin/bash
set -eu
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export TORCH_CUDA_ARCH_LIST=8.0
export NVCC_FLAGS="-Xptxas -v"
export TORCH_EXTENSIONS_DIR=/mnt/storage_pool/liaoyuanjun/higs_p2_1a_cache
export TMPDIR=/mnt/storage_pool/liaoyuanjun/tmp_p2_1a
cd /mnt/storage_pool/liaoyuanjun/higs_p2_1a_worktree/gsplat/experimental/render/kernels/cuda
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python build.py > /mnt/storage_pool/liaoyuanjun/higs_p2_1a_build_fp32.log 2>&1
echo "BUILD_EXIT=$?"