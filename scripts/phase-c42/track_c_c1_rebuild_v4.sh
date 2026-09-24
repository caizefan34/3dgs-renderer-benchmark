#!/bin/bash
# C1 rebuild: conda g++-10 + system nvcc 11.5
set -e

# Restore PyTorch cpp_extension.py from backup
CPP_EXT=/home/liaoyuanjun/.local/lib/python3.10/site-packages/torch/utils/cpp_extension.py
if [ -f "${CPP_EXT}.bak" ]; then
    cp "${CPP_EXT}.bak" "$CPP_EXT"
    echo "Restored cpp_extension.py from backup"
fi

# Use conda g++-10 with system nvcc 11.5
export CC=/home/liaoyuanjun/miniforge3/bin/gcc
export CXX=/home/liaoyuanjun/miniforge3/bin/g++
export CUDA_HOME=/usr
export PATH=/usr/bin:$PATH

echo "=== Compiler versions ==="
echo "gcc: $($CC --version 2>&1 | head -1)"
echo "g++: $($CXX --version 2>&1 | head -1)"
echo "nvcc: $(nvcc --version 2>&1 | tail -1)"

echo ""
echo "=== Rebuilding gsplat with C1 patch ==="
cd /tmp/gsplat_c1_build
rm -rf build/ *.so

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
echo "=== Running C1 benchmark ==="
cd ~/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=0 python3 scripts/phase-c42/c1_key_compression_benchmark.py 2>&1
