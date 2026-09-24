#!/bin/bash
# C17-0: Build gsplat with tile-segmented depth-only sort
# Uses conda nvcc 11.8 + conda g++-10
set -e

REPO_DIR="$HOME/3dgs-renderer-benchmark"
BUILD_DIR="/tmp/gsplat_c17_0_build"
CUDA_HOME="/home/liaoyuanjun/miniforge3"  # conda CUDA 11.8

echo "=== C17-0: Tile-Segmented Depth-Only Sort Build ==="

# Step 1: Get fresh gsplat source
echo "[1/4] Preparing gsplat source..."
rm -rf "$BUILD_DIR"
mkdir -p /tmp/gsplat_dl2
cd /tmp/gsplat_dl2
pip3 download gsplat==1.5.3 --no-deps --no-binary :all: -d /tmp/gsplat_dl2 2>/dev/null || true
tar xf gsplat-1.5.3.tar.gz 2>/dev/null
mv gsplat-1.5.3 "$BUILD_DIR"
echo "  Build dir: $BUILD_DIR"

# Step 2: Apply C17-0 patch
echo "[2/4] Applying C17-0 patch..."

# 2a: Append C17-0 functions to IntersectTile.cu (before closing namespace)
GSPLAT_SRC="$BUILD_DIR/gsplat/cuda/csrc"
C17_0_ADDITIONS="$REPO_DIR/patches/c17_0_additions.cu"

# First add ATen/Functions.h include at the top of IntersectTile.cu
/usr/bin/python3.10 -c "
with open('$GSPLAT_SRC/IntersectTile.cu') as f:
    content = f.read()
if '#include <ATen/Functions.h>' not in content:
    content = content.replace(
        '#include <ATen/core/Tensor.h>',
        '#include <ATen/core/Tensor.h>\n#include <ATen/Functions.h>'
    )
    with open('$GSPLAT_SRC/IntersectTile.cu', 'w') as f:
        f.write(content)
    print('IntersectTile.cu: added ATen/Functions.h include')
else:
    print('IntersectTile.cu: ATen/Functions.h already present')
"

# Insert the C17-0 code before the last closing namespace brace
/usr/bin/python3.10 -c "
with open('$GSPLAT_SRC/IntersectTile.cu') as f:
    content = f.read()
with open('$C17_0_ADDITIONS') as f:
    additions = f.read()
idx = content.rfind('} // namespace gsplat')
if idx == -1:
    idx = content.rfind('}  // namespace gsplat')
if idx == -1:
    print('ERROR: could not find namespace closing brace')
    exit(1)
patched = content[:idx] + additions + '\n' + content[idx:]
with open('$GSPLAT_SRC/IntersectTile.cu', 'w') as f:
    f.write(patched)
print('IntersectTile.cu patched: C17-0 functions added')
"

# 2b: Add declaration to Intersect.h
python3 -c "
with open('$GSPLAT_SRC/Intersect.h') as f:
    content = f.read()
decl = '''
void tile_segmented_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
);
'''
# Insert before closing namespace
idx = content.rfind('} // namespace gsplat')
patched = content[:idx] + decl + '\n' + content[idx:]
with open('$GSPLAT_SRC/Intersect.h', 'w') as f:
    f.write(patched)
print('Intersect.h patched: declaration added')
"

# 2c: Modify Intersect.cpp to use C17-0 when segmented=true
/usr/bin/python3.10 "$REPO_DIR/scripts/phase-c42/c17_0_patch_intersect_cpp.py" "$GSPLAT_SRC/Intersect.cpp"

# 2d: No host function needed in Intersect.cpp — everything is in IntersectTile.cu
echo "  Intersect.cpp: only function name swap (no host function appended)"

# 2e: No launch function declarations needed — they're internal to IntersectTile.cu
echo "  Intersect.h: only tile_segmented_sort declaration needed (already added)"

# Verify patch
echo "  Verifying patch..."
grep -c "tile_segmented_sort_double_buffer" "$GSPLAT_SRC/IntersectTile.cu" || true
grep -c "tile_segmented_sort_double_buffer" "$GSPLAT_SRC/Intersect.cpp" || true
grep -c "tile_segmented_sort_double_buffer" "$GSPLAT_SRC/Intersect.h" || true

# Step 3: Build
echo "[3/4] Building gsplat with C17-0 patch..."

# Use conda's nvcc 11.8 with SYSTEM gcc/g++ 11 (nvcc 11.8 should handle gcc 11)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME=/home/liaoyuanjun/miniforge3
# Put conda bin first for nvcc, but use /usr/bin/python3 explicitly
export PATH=/home/liaoyuanjun/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=/home/liaoyuanjun/miniforge3/lib:${LD_LIBRARY_PATH:-}

# Use the system python3.10 that has torch installed
PYTHON=/usr/bin/python3.10
echo "  Python: $($PYTHON --version 2>&1)"
echo "  nvcc: $(nvcc --version 2>&1 | tail -1)"
echo "  torch: $($PYTHON -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1)"

# Restore PyTorch cpp_extension.py from backup if needed
CPP_EXT=/home/liaoyuanjun/.local/lib/python3.10/site-packages/torch/utils/cpp_extension.py
if [ -f "${CPP_EXT}.bak" ]; then
    cp "${CPP_EXT}.bak" "$CPP_EXT"
fi

cd "$BUILD_DIR"
rm -rf build/ *.so

TORCH_CUDA_ARCH_LIST="8.0" $PYTHON -m pip install -e . --no-deps --no-build-isolation 2>&1 | tail -30

# Step 4: Verify build
echo "[4/4] Verifying build..."
/usr/bin/python3.10 -c "
import gsplat, os, torch, glob, time
print('gsplat version:', gsplat.__version__)
print('torch CUDA:', torch.version.cuda)
d = os.path.dirname(gsplat.__file__)
p = os.path.join(d, 'cuda', 'csrc', 'IntersectTile.cu')
with open(p) as f:
    c = f.read()
has_c17 = 'tile_segmented_sort_double_buffer' in c and 'histogram_tile_kernel' in c
print('C17-0 applied (source):', has_c17)
so_files = glob.glob(os.path.join(d, '*.so'))
if so_files:
    for so in so_files:
        mtime = os.path.getmtime(so)
        age = time.time() - mtime
        print(f'{os.path.basename(so)}: age={age:.0f}s (recent: {age < 300})')
    try:
        from gsplat import rasterization, isect_tiles
        print('Import test: PASS')
    except Exception as e:
        print(f'Import test: FAIL - {e}')
else:
    print('.so files: NOT FOUND - build failed')
" 2>&1

echo "=== Build complete ==="
