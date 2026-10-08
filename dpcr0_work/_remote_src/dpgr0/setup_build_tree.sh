#!/usr/bin/env bash
set -euo pipefail
DST=/root/epic05_dpgr0/src_dpgr0/gsplat
CUDA_DIR="$DST/experimental/render/kernels/cuda"
GI="$CUDA_DIR/csrc/gaussian_inference"

# Remove DPC-specific translation units + headers (keep C0 forward sources).
rm -f "$CUDA_DIR/ext_dpc.cpp"
rm -f "$GI/HigsNativeBackwardDpc.cu" "$GI/HigsNativeBackwardDpc.h"
rm -f "$GI/dpc_stage1.cu" "$GI/dpc_stage1.h"

# Generate build_dpgr0.py from build_dpc.py.
cp "$CUDA_DIR/build_dpc.py" "$CUDA_DIR/build_dpgr0.py"
sed -i 's/dpc0_gaussian_render_inference_scene_cuda/dpgr0_gaussian_render_inference_scene_cuda/g' "$CUDA_DIR/build_dpgr0.py"
sed -i 's/ext_dpc\.cpp/ext_dpgr0.cpp/g' "$CUDA_DIR/build_dpgr0.py"
sed -i 's/HigsNativeBackwardDpc\.cu/HigsNativeBackwardDpgr0.cu/g' "$CUDA_DIR/build_dpgr0.py"
sed -i 's/dpc_stage1\.cu/dpgr0_stage1.cu/g' "$CUDA_DIR/build_dpgr0.py"
sed -i 's/build_and_load_dpc0/build_and_load_dpgr0/g' "$CUDA_DIR/build_dpgr0.py"

echo "=== build_dpgr0.py key lines ==="
grep -n 'name =\|ext_dpgr0\|HigsNativeBackwardDpgr0\|dpgr0_stage1\|build_and_load_dpgr0' "$CUDA_DIR/build_dpgr0.py"
echo "=== residual DPC refs in cuda dir (excluding build_dpgr0.py) ==="
if grep -rn 'dpc_stage1\|HigsNativeBackwardDpc\|ext_dpc' "$CUDA_DIR" --include=*.py --include=*.cpp --include=*.cu --include=*.h | grep -v build_dpgr0.py; then
  echo "FOUND_RESIDUAL"
else
  echo NONE
fi
echo "=== gaussian_inference dir now ==="
ls "$GI"
