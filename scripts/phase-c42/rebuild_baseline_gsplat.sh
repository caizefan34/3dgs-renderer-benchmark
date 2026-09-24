#!/bin/bash
# Rebuild baseline gsplat with CUDA 11.8 from source
set -e

export CUDA_HOME=/home/liaoyuanjun/miniforge3
export PATH=/usr/bin:/home/liaoyuanjun/miniforge3/bin:$PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=8.0

echo "python: $(python3 --version 2>&1)"
echo "torch: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1)"
echo "nvcc: $(nvcc --version 2>&1 | tail -1)"

# Use pip download + manual build to avoid conda python
pip3 download gsplat==1.5.3 --no-deps --no-binary :all: -d /tmp/gsplat_baseline 2>/dev/null || true
cd /tmp/gsplat_baseline
rm -rf gsplat-1.5.3
tar xf gsplat-1.5.3.tar.gz
cd gsplat-1.5.3

# Build with --no-build-isolation so it uses system python's torch
python3 -m pip install -e . --no-deps --no-build-isolation 2>&1 | tail -20

# Verify
python3 -c "
import gsplat, torch
from gsplat import fully_fused_projection, isect_tiles
print('gsplat:', gsplat.__version__)
print('torch CUDA:', torch.version.cuda)
means2d = torch.randn(100, 2, device='cuda')
radii = torch.ones(100, 2, device='cuda', dtype=torch.int32) * 10
depths = torch.randn(100, device='cuda')
tpg, iids, fids = isect_tiles(means2d, radii, depths, 16, 120, 68, sort=True, packed=False)
print(f'isect_tiles test: PASS (n_isects={len(fids)})')
" 2>&1
