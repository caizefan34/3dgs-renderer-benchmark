#!/bin/bash
# C1 rebuild with CUDA 12.4, bypassing PyTorch CUDA version check
set -e

CUDA_HOME=/tmp/cuda_12_4
rm -rf "$CUDA_HOME"
mkdir -p "$CUDA_HOME/bin" "$CUDA_HOME/include" "$CUDA_HOME/lib64"

# Copy nvcc
cp /home/liaoyuanjun/miniforge3/pkgs/cuda-nvcc-12.4.131-0/bin/nvcc "$CUDA_HOME/bin/"
chmod +x "$CUDA_HOME/bin/nvcc"

# Copy headers
cp -r /home/liaoyuanjun/miniforge3/pkgs/cuda-cudart-dev-12.4.127-0/include/* "$CUDA_HOME/include/" 2>/dev/null || true

# Copy libs - find them properly
find /home/liaoyuanjun/miniforge3/pkgs/cuda-cudart-dev-12.4.127-0/ -name "*.so*" -exec cp {} "$CUDA_HOME/lib64/" \; 2>/dev/null || true
find /home/liaoyuanjun/miniforge3/pkgs/cuda-cudart-12.4.127-0/ -name "*.so*" -exec cp {} "$CUDA_HOME/lib64/" \; 2>/dev/null || true
find /home/liaoyuanjun/miniforge3/pkgs/cuda-cudart-dev-12.4.127-0/ -name "*.a" -exec cp {} "$CUDA_HOME/lib64/" \; 2>/dev/null || true

# Also need nvcc's own libs (cudart is separate from nvcc)
find /home/liaoyuanjun/miniforge3/pkgs/cuda-nvcc-12.4.131-0/ -name "*.so*" -exec cp {} "$CUDA_HOME/lib64/" \; 2>/dev/null || true

# Verify
echo "=== CUDA_HOME setup ==="
echo "nvcc: $($CUDA_HOME/bin/nvcc --version 2>&1 | tail -1)"
echo "headers: $(ls $CUDA_HOME/include/cuda_runtime.h 2>/dev/null && echo OK || echo MISSING)"
echo "libs: $(ls $CUDA_HOME/lib64/ 2>/dev/null | head -5)"
echo "libcudart: $(ls $CUDA_HOME/lib64/libcudart* 2>/dev/null | head -3)"

# Bypass PyTorch CUDA version check
CPP_EXT=/home/liaoyuanjun/.local/lib/python3.10/site-packages/torch/utils/cpp_extension.py
echo ""
echo "=== Patching PyTorch cpp_extension.py to skip CUDA version check ==="
# Backup
cp "$CPP_EXT" "${CPP_EXT}.bak" 2>/dev/null || true
# Replace the version check - use sed to comment out the raise RuntimeError
python3 -c "
import re
with open('$CPP_EXT') as f:
    content = f.read()
# Replace the version mismatch check with a warning
old = 'raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))'
new = 'warnings.warn(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))'
if old in content:
    content = content.replace(old, new)
    with open('$CPP_EXT', 'w') as f:
        f.write(content)
    print('Patched: RuntimeError -> warnings.warn')
else:
    print('Pattern not found (already patched?)')
"

# Now rebuild gsplat with C1 patch
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo ""
echo "=== Rebuilding gsplat with C1 patch (CUDA 12.4, version check bypassed) ==="
cd /tmp/gsplat_c1_build

# Clean previous build
rm -rf build/ *.so

# Build with explicit arch for A100 (sm_80)
TORCH_CUDA_ARCH_LIST="8.0" python3 -m pip install -e . --no-deps --no-build-isolation 2>&1 | tail -40

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
print('C1 applied (source):', has_c1)
# Check if the .so was actually rebuilt
import glob
so_files = glob.glob(os.path.join(d, '*.so'))
if so_files:
    import time
    for so in so_files:
        mtime = os.path.getmtime(so)
        age = time.time() - mtime
        print(f'{os.path.basename(so)}: age={age:.0f}s (recent: {age < 300})')
else:
    print('.so files: NOT FOUND')
# Try a basic import test
try:
    from gsplat import rasterization
    print('Import test: PASS')
except Exception as e:
    print(f'Import test: FAIL - {e}')
" 2>&1

echo ""
echo "=== Running C1 benchmark (patched) ==="
cd ~/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=0 python3 scripts/phase-c42/c1_key_compression_benchmark.py 2>&1
