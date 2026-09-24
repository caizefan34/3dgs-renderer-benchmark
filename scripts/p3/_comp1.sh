#!/bin/bash
# Compile only HigsNativeBackward.cu and print the FIRST nvcc errors (root cause).
cd /mnt/storage_pool/liaoyuanjun/higs_p3h_cache/V0/experimental_gaussian_render_inference_scene_cuda
export PATH=/usr/bin:/bin:/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/nvcc \
  --generate-dependencies-with-compile --dependency-output HigsNativeBackward.cuda.o.d \
  -DTORCH_EXTENSION_NAME=experimental_gaussian_render_inference_scene_cuda \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  -I/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/cuda/include \
  -I/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/cuda/csrc \
  -I/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference \
  -I/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/cuda/csrc/third_party/glm \
  -I/mnt/storage_pool/liaoyuanjun/higs-13scene-env/targets/x86_64-linux/include \
  -isystem /mnt/storage_pool/liaoyuanjun/higs-13scene-env/lib/python3.10/site-packages/torch/include \
  -isystem /mnt/storage_pool/liaoyuanjun/higs-13scene-env/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
  -isystem /mnt/storage_pool/liaoyuanjun/higs-13scene-env/include \
  -isystem /mnt/storage_pool/liaoyuanjun/higs-13scene-env/include/python3.10 \
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
  -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 \
  --compiler-options '-fPIC' --forward-unknown-opts -use_fast_math \
  -diag-suppress 20012,186 --expt-relaxed-constexpr -std=c++20 -O3 -DNDEBUG \
  -Wno-attributes -Wno-unknown-pragmas -DAT_PARALLEL_OPENMP -fopenmp \
  -DP3_SINK_MODE=0 \
  -c /mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu \
  -o HigsNativeBackward.cuda.o 2>&1 | grep -n 'error' | head -30
echo "EXIT_DONE"