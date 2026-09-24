#!/bin/bash
# C1 rebuild: conda g++-10 + system CUDA 11.5 headers
set -e

# Restore PyTorch cpp_extension.py from backup
CPP_EXT=/home/liaoyuanjun/.local/lib/python3.10/site-packages/torch/utils/cpp_extension.py
if [ -f "${CPP_EXT}.bak" ]; then
    cp "${CPP_EXT}.bak" "$CPP_EXT"
    echo "Restored cpp_extension.py from backup"
fi

# Use conda g++-10 with system CUDA 11.5
export CC=/home/liaoyuanjun/miniforge3/bin/gcc
export CXX=/home/liaoyuanjun/miniforge3/bin/g++
export CUDA_HOME=/usr

# Add system include paths for CUDA headers
export CFLAGS="-I/usr/include"
export CXXFLAGS="-I/usr/include"
export CPPFLAGS="-I/usr/include"

echo "=== Compiler versions ==="
echo "gcc: $($CC --version 2>&1 | head -1)"
echo "g++: $($CXX --version 2>&1 | head -1)"
echo "nvcc: $(nvcc --version 2>&1 | tail -1)"
echo "cuda_runtime_api.h: $(ls /usr/include/cuda_runtime_api.h 2>/dev/null && echo OK || echo MISSING)"

echo ""
echo "=== Rebuilding gsplat with C1 patch ==="
cd /tmp/gsplat_c1_build
rm -rf build/ *.so

# Build - need to add /usr/include to the include path
# PyTorch's cpp_extension uses CUDA_HOME/include for CUDA headers
# But CUDA_HOME=/usr means it looks for /usr/include which IS correct
# The issue is that conda gcc has different default include paths

# Let's create a symlink structure
CUDA_SYMLINK=/tmp/cuda_115
rm -rf "$CUDA_SYMLINK"
mkdir -p "$CUDA_SYMLINK/bin" "$CUDA_SYMLINK/include" "$CUDA_SYMLINK/lib64"

ln -sf /usr/bin/nvcc "$CUDA_SYMLINK/bin/nvcc"
ln -sf /usr/include/cuda_runtime.h "$CUDA_SYMLINK/include/cuda_runtime.h"
ln -sf /usr/include/cuda_runtime_api.h "$CUDA_SYMLINK/include/cuda_runtime_api.h"
# Link all CUDA headers
for h in /usr/include/cuda*.h /usr/include/nv*.h; do
    [ -f "$h" ] && ln -sf "$h" "$CUDA_SYMLINK/include/$(basename $h)" 2>/dev/null
done
# Link CUDA sub-headers
for d in /usr/include/cuda; do
    [ -d "$d" ] && ln -sf "$d" "$CUDA_SYMLINK/include/cuda" 2>/dev/null
done
# Link libs
for l in /usr/lib/x86_64-linux-gnu/libcudart*; do
    [ -f "$l" ] && ln -sf "$l" "$CUDA_SYMLINK/lib64/$(basename $l)" 2>/dev/null
done

echo "CUDA symlink: $(ls $CUDA_SYMLINK/include/cuda_runtime_api.h 2>/dev/null && echo OK || echo MISSING)"

export CUDA_HOME="$CUDA_SYMLINK"
export PATH="$CUDA_SYMLINK/bin:$PATH"

TORCH_CUDA_ARCH_LIST="8.0" python3 -m pip install -e . --no-deps --no-build-isolation 2>&1 | tail -40

echo ""
echo "=== Build result ==="
python3 -c "
import gsplat, os, torch, glob, time
print('gsplat version:', gsplat.__version__)
print('torch CUDA:', torch.version.cuda)
d = os.path.dirname(gsplat.__file__)
p = os.path.join(d, 'cuda', 'csrc', 'IntersectTile.cu')
with open(p) as f:
    c = f.read()
has_c1 = 'depth_upper' in c and '16 + tile_n_bits' in c
print('C1 applied (source):', has_c1)
so_files = glob.glob(os.path.join(d, '*.so'))
if so_files:
    for so in so_files:
        mtime = os.path.getmtime(so)
        age = time.time() - mtime
        print(f'{os.path.basename(so)}: age={age:.0f}s (recent: {age < 300})')
    try:
        from gsplat import rasterization
        print('Import test: PASS')
    except Exception as e:
        print(f'Import test: FAIL - {e}')
else:
    print('.so files: NOT FOUND - build failed')
" 2>&1

echo ""
echo "=== Running C1 benchmark (with C1 patch if build succeeded) ==="
cd ~/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=0 python3 scripts/phase-c42/c1_key_compression_benchmark.py 2>&1
