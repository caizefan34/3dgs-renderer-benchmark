#!/bin/bash
# C1 rebuild: install g++-10 via conda, use with system nvcc 11.5
set -e

echo "=== Step 1: Install g++-10 via conda ==="
# Install gcc/g++ 10 via conda-forge (doesn't affect Python)
mamba install -c conda-forge gxx_linux-64=10 gcc_linux-64=10 -y 2>&1 | tail -10 || \
conda install -c conda-forge gxx_linux-64=10 gcc_linux-64=10 -y 2>&1 | tail -10

# Find the conda g++-10
CONDA_GXX=$(find /home/liaoyuanjun/miniforge3 -name "x86_64-conda-linux-gnu-g++" -path "*10*" 2>/dev/null | head -1)
if [ -z "$CONDA_GXX" ]; then
    CONDA_GXX=$(find /home/liaoyuanjun/miniforge3 -name "x86_64-conda-linux-gnu-g++" 2>/dev/null | head -1)
fi
echo "Conda g++: $CONDA_GXX"
if [ -n "$CONDA_GXX" ]; then
    $CONDA_GXX --version 2>&1 | head -1
fi

CONDA_GCC=$(echo "$CONDA_GXX" | sed 's/g++/gcc/')
echo "Conda gcc: $CONDA_GCC"

# Also find the actual g++-10 binary
GXX10=$(find /home/liaoyuanjun/miniforge3 -name "g++-10" 2>/dev/null | head -1)
GCC10=$(find /home/liaoyuanjun/miniforge3 -name "gcc-10" 2>/dev/null | head -1)
echo "Direct g++-10: $GXX10"
echo "Direct gcc-10: $GCC10"

if [ -z "$GXX10" ] && [ -z "$CONDA_GXX" ]; then
    echo "ERROR: No g++-10 found. Trying alternative..."
    # Try to find any g++ that's not version 11
    for ver in 10 9 12; do
        GXX=$(find /home/liaoyuanjun/miniforge3 -name "g++-$ver" 2>/dev/null | head -1)
        if [ -n "$GXX" ]; then
            GXX10="$GXX"
            GCC10=$(find /home/liaoyuanjun/miniforge3 -name "gcc-$ver" 2>/dev/null | head -1)
            echo "Found g++-$ver: $GXX10"
            break
        fi
    done
fi

if [ -z "$GXX10" ] && [ -z "$CONDA_GXX" ]; then
    echo "FATAL: No compatible g++ found. Falling back to estimation."
    exit 1
fi

# Use the best available compiler
if [ -n "$CONDA_GXX" ]; then
    export CXX="$CONDA_GXX"
    export CC="$CONDA_GCC"
elif [ -n "$GXX10" ]; then
    export CXX="$GXX10"
    export CC="$GCC10"
fi

echo ""
echo "=== Step 2: Rebuild gsplat with C1 patch ==="
echo "CC=$CC"
echo "CXX=$CXX"

# Restore PyTorch cpp_extension.py from backup
CPP_EXT=/home/liaoyuanjun/.local/lib/python3.10/site-packages/torch/utils/cpp_extension.py
if [ -f "${CPP_EXT}.bak" ]; then
    cp "${CPP_EXT}.bak" "$CPP_EXT"
    echo "Restored cpp_extension.py from backup"
fi

# Use system CUDA 11.5 (nvcc) with conda g++-10
export CUDA_HOME=/usr  # system cuda
export PATH=/usr/bin:$PATH  # system nvcc 11.5

# Clean and rebuild
cd /tmp/gsplat_c1_build
rm -rf build/ *.so

# Build with g++-10 and nvcc 11.5
export CC="$CC"
export CXX="$CXX"
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
echo "=== Step 3: Running C1 benchmark (if build succeeded) ==="
cd ~/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=0 python3 scripts/phase-c42/c1_key_compression_benchmark.py 2>&1
