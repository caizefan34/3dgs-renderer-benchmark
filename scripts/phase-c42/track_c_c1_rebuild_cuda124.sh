#!/bin/bash
# C1 rebuild with CUDA 12.4 from miniforge
set -e

CUDA_HOME=/tmp/cuda_12_4
rm -rf "$CUDA_HOME"
mkdir -p "$CUDA_HOME/bin" "$CUDA_HOME/include" "$CUDA_HOME/lib64"

# Copy nvcc from miniforge
cp /home/liaoyuanjun/miniforge3/pkgs/cuda-nvcc-12.4.131-0/bin/nvcc "$CUDA_HOME/bin/"
chmod +x "$CUDA_HOME/bin/nvcc"

# Copy headers from miniforge cudart
cp -r /home/liaoyuanjun/miniforge3/pkgs/cuda-cudart-dev-12.4.127-0/include/* "$CUDA_HOME/include/" 2>/dev/null || true

# Copy libs
cp /home/liaoyuanjun/miniforge3/pkgs/cuda-cudart-dev-12.4.127-0/lib/* "$CUDA_HOME/lib64/" 2>/dev/null || true
cp /home/liaoyuanjun/miniforge3/pkgs/cuda-cudart-12.4.127-0/lib/*.so* "$CUDA_HOME/lib64/" 2>/dev/null || true

# Verify
echo "=== CUDA_HOME setup ==="
echo "nvcc: $($CUDA_HOME/bin/nvcc --version 2>&1 | tail -1)"
echo "headers: $(ls $CUDA_HOME/include/cuda_runtime.h 2>/dev/null && echo OK || echo MISSING)"
echo "libs: $(ls $CUDA_HOME/lib64/libcudart.so* 2>/dev/null | head -1 || echo MISSING)"

# Now rebuild gsplat with C1 patch
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo ""
echo "=== Rebuilding gsplat with C1 patch (CUDA 12.4) ==="
cd /tmp/gsplat_c1_build

# Clean previous build
rm -rf build/ *.so

# Build
TORCH_CUDA_ARCH_LIST="8.0" python3 -m pip install -e . --no-deps --no-build-isolation 2>&1 | tail -30

echo ""
echo "=== Build result ==="
python3 -c "
import gsplat, os, torch
print('gsplat version:', gsplat.__version__)
print('torch CUDA:', torch.version.cuda)
d = os.path.dirname(gsplat.__file__)
p = os.path.join(d, 'cuda', 'csrc', 'IntersectTile.cu')
with open(p) as f:
    c = f.read()
has_c1 = 'depth_upper' in c and '16 + tile_n_bits' in c
print('C1 applied:', has_c1)
# Check if the .so was actually rebuilt
so_path = os.path.join(d, '_C.so')
if os.path.exists(so_path):
    import time
    mtime = os.path.getmtime(so_path)
    age = time.time() - mtime
    print(f'_C.so age: {age:.0f}s (rebuilt recently: {age < 300})')
else:
    print('_C.so: NOT FOUND')
" 2>&1

echo ""
echo "=== Running C1 benchmark ==="
cd ~/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=0 python3 scripts/phase-c42/c1_key_compression_benchmark.py 2>&1
