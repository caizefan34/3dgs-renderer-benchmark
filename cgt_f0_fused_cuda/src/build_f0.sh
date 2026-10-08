#!/usr/bin/env bash
# Build the F0 fused CUDA shared library with the CUDA 11.8 toolchain (matches torch cu118).
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NVCC="/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118/bin/nvcc"
EXTRA="$1"
echo "NVCC=$NVCC EXTRA=$EXTRA"
"$NVCC" -std=c++17 -O3 -arch=sm_80 -Xcompiler -fPIC -Xptxas -v \
    $EXTRA --shared -o "$HERE/f0_kernels.so" "$HERE/f0_kernels.cu" 2> "$HERE/ptxas.log"
echo "BUILD_OK -> $HERE/f0_kernels.so"
echo "--- ptxas (registers/shared) ---"
grep -E "registers|spill|shared|bytes" "$HERE/ptxas.log" | head -60 || true
