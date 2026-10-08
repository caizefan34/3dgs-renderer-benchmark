#!/bin/bash
# DPC0 build: mirror the frozen C0 extension build, swap in the DPC backward.
# Produces a SEPARATE .so; never touches the authoritative C0 .so.
set -e
SRC=/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda
GSI=$SRC/csrc/gaussian_inference
DPC=/root/epic05_dpc0/dpc_src
BLD=/root/epic05_dpc0/build_dpc
EXT=$BLD/experimental_gaussian_render_inference_scene_cuda
NVCC=/root/miniforge3/envs/gsplat/bin/nvcc
CXX=c++
TORCH_INC=/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/include
TORCH_INC2=/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/include/torch/csrc/api/include
PY_INC=/root/miniforge3/envs/gsplat/include/python3.10

rm -rf $BLD
mkdir -p $EXT
# C0 sources except HigsNativeBackward.cu
cp /root/epic05_dpc0/dpc_src/ext_dpc.cpp $EXT/
for f in GaussianRenderInferenceScene.cu GatherVisible.cu GatherlessProjectedProducer.cu \
         IntersectCommon.cu IntersectMTFused.cu MacroTileIntersect.cu MacroTileRasterize.cu \
         Projection.cu SegmentedSort.cu SHCompression.cu SphericalHarmonics.cu; do
  cp $GSI/$f $EXT/
done
# DPC backward (self-contained: projection/SH/reduce kernels + dpc launcher)
cp $DPC/HigsNativeBackwardDpc.cu $EXT/
cp $DPC/dpc_stage1.cu $EXT/
cp $DPC/dpc_stage1.h $EXT/
cp $DPC/HigsNativeBackwardDpc.h $EXT/
cp $GSI/HigsNativeBackward.h $EXT/ 2>/dev/null || true
cp $GSI/Config.h $GSI/Common.h $GSI/RasterizeToPixels3DGSDevice.cuh $GSI/Utils.cuh $EXT/ 2>/dev/null || true
echo "=== build dir ready ==="
ls $EXT/

INC="-I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/include \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/csrc \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/csrc/third_party/glm \
 -I/root/miniforge3/envs/gsplat/targets/x86_64-linux-gnu/include \
 -isystem $TORCH_INC -isystem $TORCH_INC2 -isystem /root/miniforge3/envs/gsplat/include \
 -I$EXT"

CUDA_FLAGS="-DTORCH_EXTENSION_NAME=experimental_gaussian_render_inference_scene_cuda -DTORCH_API_INCLUDE_EXTENSION_H \
 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
 --expt-relaxed-constexpr -gencode=arch=compute_80,code=sm_80 --compiler-options '-fPIC' --forward-unknown-opts \
 -use_fast_math -diag-suppress 20012,186 --expt-relaxed-constexpr -std=c++20 -O3 -DNDEBUG \
 -Wno-attributes -Wno-unknown-pragmas -DAT_PARALLEL_OPENMP -fopenmp"

cd $EXT
echo "=== compiling DPC backward (HigsNativeBackwardDpc.cu) ==="
$NVCC $INC $CUDA_FLAGS -c HigsNativeBackwardDpc.cu -o HigsNativeBackwardDpc.cuda.o 2> HigsNativeBackwardDpc.compile.log || {
  echo "COMPILE FAIL HigsNativeBackwardDpc.cu"; tail -40 HigsNativeBackwardDpc.compile.log; exit 1; }
echo "OK HigsNativeBackwardDpc.cu"
echo "=== compiling dpc_stage1.cu ==="
$NVCC $INC $CUDA_FLAGS -c dpc_stage1.cu -o dpc_stage1.cuda.o 2> dpc_stage1.compile.log || {
  echo "COMPILE FAIL dpc_stage1.cu"; tail -40 dpc_stage1.compile.log; exit 1; }
echo "OK dpc_stage1.cu"
echo "DONE-PHASE1"
