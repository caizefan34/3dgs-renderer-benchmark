#!/bin/bash
# DPGR0 build: mirror the frozen C0/DPC0 extension build, swap in the DPGR0
# direct-gather Stage1. Produces a SEPARATE .so; never touches C0/DPC0 .so.
# .cu files compile with nvcc; ext_dpgr0.cpp (host pybind) compiles with c++.
set -e
SRC=/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda
GSI=$SRC/csrc/gaussian_inference
BLD=/root/epic05_dpgr0/build_dpgr0
EXT=$BLD/experimental_gaussian_render_inference_scene_cuda
NVCC=/root/miniforge3/envs/gsplat/bin/nvcc
CXX=c++
TORCH_LIB=/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/lib
TORCH_INC=/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/include
TORCH_INC2=/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/include/torch/csrc/api/include
PY_INC=/root/miniforge3/envs/gsplat/include/python3.10
CUDA_INC=/usr/local/cuda/include

rm -rf $BLD
mkdir -p $EXT
cp /root/epic05_dpgr0/src/ext_dpgr0.cpp $EXT/
for f in GaussianRenderInferenceScene.cu GatherVisible.cu GatherlessProjectedProducer.cu \
         IntersectCommon.cu IntersectMTFused.cu MacroTileIntersect.cu MacroTileRasterize.cu \
         Projection.cu SegmentedSort.cu SHCompression.cu SphericalHarmonics.cu; do
  cp $GSI/$f $EXT/
done
cp /root/epic05_dpgr0/src/HigsNativeBackwardDpgr0.cu $EXT/
cp /root/epic05_dpgr0/src/dpgr0_stage1.cu $EXT/
cp /root/epic05_dpgr0/src/dpgr0_stage1.h $EXT/
cp $GSI/HigsNativeBackward.h $EXT/
cp $GSI/Config.h $GSI/Common.h $GSI/RasterizeToPixels3DGSDevice.cuh $GSI/Utils.cuh $GSI/Utils.h $EXT/ 2>/dev/null || true
echo "=== build dir ready ==="

INC="-I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/include \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/csrc \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/csrc/third_party/glm \
 -I/root/miniforge3/envs/gsplat/targets/x86_64-linux-gnu/include \
 -isystem $TORCH_INC -isystem $TORCH_INC2 -isystem $CUDA_INC -isystem $PY_INC \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda -I$EXT"

CUDA_FLAGS="-DTORCH_EXTENSION_NAME=experimental_gaussian_render_inference_scene_cuda -DTORCH_API_INCLUDE_EXTENSION_H \
 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
 --expt-relaxed-constexpr -gencode=arch=compute_80,code=sm_80 --compiler-options '-fPIC' --forward-unknown-opts \
 -use_fast_math -diag-suppress 20012,186 --expt-relaxed-constexpr -std=c++20 -O3 -DNDEBUG \
 -Wno-attributes -Wno-unknown-pragmas -DAT_PARALLEL_OPENMP -fopenmp"

CXX_FLAGS="-DTORCH_EXTENSION_NAME=experimental_gaussian_render_inference_scene_cuda -DTORCH_API_INCLUDE_EXTENSION_H \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/include \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/csrc \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/cuda/csrc/third_party/glm \
 -isystem $TORCH_INC -isystem $TORCH_INC2 -isystem $CUDA_INC -isystem $PY_INC \
 -I/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda -fPIC -std=c++17 -std=c++20 -O3 -DNDEBUG -Wno-attributes -Wno-unknown-pragmas \
 -DAT_PARALLEL_OPENMP -fopenmp"

cd $EXT
compile_cu() {
  local src="$1" obj="$2" log="$3"
  $NVCC $INC $CUDA_FLAGS -c "$src" -o "$obj" 2> "$log" || { echo "COMPILE FAIL $src"; tail -60 "$log"; exit 1; }
  echo "OK $src"
}

echo "=== compiling ext_dpgr0.cpp (host c++) ==="
$CXX $CXX_FLAGS -c ext_dpgr0.cpp -o ext_dpgr0.cuda.o 2> ext_dpgr0.compile.log || { echo "COMPILE FAIL ext_dpgr0.cpp"; tail -60 ext_dpgr0.compile.log; exit 1; }
echo "OK ext_dpgr0.cpp"
echo "=== compiling DPGR0 backward (HigsNativeBackwardDpgr0.cu) ==="
compile_cu HigsNativeBackwardDpgr0.cu HigsNativeBackwardDpgr0.cuda.o HigsNativeBackwardDpgr0.compile.log
echo "=== compiling dpgr0_stage1.cu ==="
compile_cu dpgr0_stage1.cu dpgr0_stage1.cuda.o dpgr0_stage1.compile.log
echo "=== compiling C0 sources ==="
for f in GaussianRenderInferenceScene.cu GatherVisible.cu GatherlessProjectedProducer.cu \
         IntersectCommon.cu IntersectMTFused.cu MacroTileIntersect.cu MacroTileRasterize.cu \
         Projection.cu SegmentedSort.cu SHCompression.cu SphericalHarmonics.cu; do
  compile_cu "$f" "${f%.cu}.cuda.o" "${f%.cu}.compile.log"
done
echo "=== linking ==="
$CXX -shared -s \
  ext_dpgr0.cuda.o HigsNativeBackwardDpgr0.cuda.o dpgr0_stage1.cuda.o \
  GaussianRenderInferenceScene.cuda.o GatherVisible.cuda.o GatherlessProjectedProducer.cuda.o \
  IntersectCommon.cuda.o IntersectMTFused.cuda.o MacroTileIntersect.cuda.o MacroTileRasterize.cuda.o \
  Projection.cuda.o SegmentedSort.cuda.o SHCompression.cuda.o SphericalHarmonics.cuda.o \
  -L$TORCH_LIB -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python \
  -L/usr/local/cuda/lib64 -lcudart \
  -o $BLD/experimental_gaussian_render_inference_scene_cuda.so \
  2> link.log || { echo "LINK FAIL"; tail -50 link.log; exit 1; }
echo "BUILT: $BLD/experimental_gaussian_render_inference_scene_cuda.so"
sha256sum $BLD/experimental_gaussian_render_inference_scene_cuda.so
