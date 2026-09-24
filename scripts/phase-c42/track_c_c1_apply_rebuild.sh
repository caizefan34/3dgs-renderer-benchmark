#!/bin/bash
# Track C: C1 Key Compression Patch Apply + Rebuild + Benchmark
# Runs on A100. Creates a temporary gsplat build with C1 patch applied.
set -e

GSPLAT_SRC_DIR="/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat"
BUILD_DIR="/tmp/gsplat_c1_build"
REPO_DIR="$HOME/3dgs-renderer-benchmark"

echo "=== C1 Key Compression Patch Apply + Rebuild ==="

# Step 1: Get gsplat source
echo "[1/5] Downloading gsplat 1.5.3 source..."
if [ ! -d "$BUILD_DIR" ]; then
    pip3 download gsplat==1.5.3 --no-deps --no-binary :all: -d /tmp/gsplat_dl 2>/dev/null || true
    mkdir -p /tmp/gsplat_dl
    cd /tmp/gsplat_dl
    # Try pip download, fallback to git clone
    pip3 download gsplat==1.5.3 --no-deps --no-binary :all: -d /tmp/gsplat_dl 2>/dev/null || true
    # Extract
    tar xf gsplat-1.5.3.tar.gz 2>/dev/null || true
    if [ ! -d "gsplat-1.5.3" ]; then
        echo "  pip download failed, trying git clone..."
        git clone --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat_dl/gsplat-1.5.3 2>/dev/null || true
    fi
    mv /tmp/gsplat_dl/gsplat-1.5.3 "$BUILD_DIR"
fi
echo "  Build dir: $BUILD_DIR"

# Step 2: Apply C1 patch
echo "[2/5] Applying C1 patch..."
cp "$REPO_DIR/patches/IntersectTile.c1.cu" "$BUILD_DIR/gsplat/cuda/csrc/IntersectTile.cu"
echo "  Patched IntersectTile.cu"

# Step 3: Build
echo "[3/5] Building gsplat with C1 patch..."
cd "$BUILD_DIR"
# Build in-place (editable install)
python3 -m pip install -e . --no-deps --no-build-isolation 2>&1 | tail -5
echo "  Build complete"

# Step 4: Verify
echo "[4/5] Verifying C1 patch..."
python3 -c "
import gsplat, os
d = os.path.dirname(gsplat.__file__)
p = os.path.join(d, 'cuda', 'csrc', 'IntersectTile.cu')
with open(p) as f:
    c = f.read()
has_c1 = 'depth_upper' in c and '16 + tile_n_bits' in c
print(f'gsplat dir: {d}')
print(f'C1 applied: {has_c1}')
print(f'gsplat version: {gsplat.__version__}')
"

# Step 5: Benchmark
echo "[5/5] Running C1 benchmark..."
cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES=4 python3 scripts/phase-c42/c1_key_compression_benchmark.py 2>&1

echo "=== C1 Benchmark Complete ==="
