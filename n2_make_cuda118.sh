#!/bin/bash
set -e

# ====================================================================
# Construct a combined CUDA 11.8 HOME from conda packages
# ====================================================================

CONDA_NVCC="$HOME/miniforge3/pkgs/cuda-nvcc-11.8.89-0"
CONDA_CUDART="$HOME/miniforge3/pkgs/cuda-cudart-dev-11.8.89-0"
CONDA_CCCL="$HOME/miniforge3/pkgs/cuda-cccl-11.8.89-0"

CUDA_HOME="/tmp/cuda118_combined"
rm -rf "$CUDA_HOME"
mkdir -p "$CUDA_HOME/bin" "$CUDA_HOME/include" "$CUDA_HOME/lib64"

# Copy/link nvcc and tools
for f in "$CONDA_NVCC/bin/"*; do
    ln -sf "$f" "$CUDA_HOME/bin/$(basename $f)"
done

# Copy cudart headers
if [ -d "$CONDA_CUDART/include" ]; then
    for f in "$CONDA_CUDART/include/"*; do
        ln -sf "$f" "$CUDA_HOME/include/$(basename $f)" 2>/dev/null || true
    done
fi

# Copy cudart libs
if [ -d "$CONDA_CUDART/lib" ]; then
    for f in "$CONDA_CUDART/lib/"*; do
        ln -sf "$f" "$CUDA_HOME/lib64/$(basename $f)" 2>/dev/null || true
    done
fi
if [ -d "$CONDA_CUDART/lib64" ]; then
    for f in "$CONDA_CUDART/lib64/"*; do
        ln -sf "$f" "$CUDA_HOME/lib64/$(basename $f)" 2>/dev/null || true
    done
fi

# Copy CCCL headers (thrust, cub, libcu++)
if [ -d "$CONDA_CCCL/include" ]; then
    cp -rs "$CONDA_CCCL/include/"* "$CUDA_HOME/include/" 2>/dev/null || true
fi

# Also need crt headers from nvcc package
if [ -d "$CONDA_NVCC/bin/crt" ]; then
    mkdir -p "$CUDA_HOME/bin/crt"
    ln -sf "$CONDA_NVCC/bin/crt/"* "$CUDA_HOME/bin/crt/" 2>/dev/null || true
fi

# Check if there's a nvcc.profile that needs fixing
cat "$CONDA_NVCC/bin/nvcc.profile" 2>/dev/null
echo "==="

# Test nvcc with the combined CUDA_HOME
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=$CUDA_HOME"
nvcc --version 2>&1 | grep release

# Test compilation with torch headers
echo '__global__ void k(){} int main(){return 0;}' > /tmp/test_cuda118.cu
nvcc -O2 -std=c++17 --use_fast_math /tmp/test_cuda118.cu -o /tmp/test_cuda118 2>&1 && echo "CUDA 11.8 basic compile OK" || echo "CUDA 11.8 basic compile FAILED"

echo "=== Verify CUDA_HOME structure ==="
ls "$CUDA_HOME/bin/nvcc" && echo "nvcc OK"
ls "$CUDA_HOME/include/cuda_runtime.h" && echo "cuda_runtime.h OK"
ls "$CUDA_HOME/include/cuda.h" && echo "cuda.h OK"
find "$CUDA_HOME/lib64" -name 'libcudart*' 2>/dev/null | head

echo "=== DONE ==="
