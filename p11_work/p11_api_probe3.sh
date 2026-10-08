#!/bin/bash
set -u
T=/root/epic05_p8/gsplat-main-512d366b
echo "=== gsplat __init__ exports ==="
grep -n 'rasterization' $T/gsplat/__init__.py | head -10
echo "=== autograd wiring ==="
grep -rn 'class.*torch.autograd.Function\|class.*autograd.Function' $T/gsplat/rendering.py $T/gsplat/cuda/_wrapper.py | head
grep -n 'def rasterization' $T/gsplat/rendering.py
sed -n '754,775p' $T/gsplat/rendering.py
echo "=== where is rasterization autograd class ==="
grep -rn 'class _Rasterization\|class Rasterization' $T/gsplat/ 2>/dev/null | grep -v __pycache__ | head
echo "=== wrapper rasterization_3dgs bwd ==="
grep -n 'rasterization_3dgs_bwd\|def rasterization_3dgs' $T/gsplat/cuda/_wrapper.py | head
echo "=== bwd kernel traversal semantics ==="
grep -n 'last_id\|current_idx\|tile_idx\|for (int32_t' $T/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu 2>/dev/null | head -30
ls $T/gsplat/cuda/csrc/ | grep -i 'RasterizeToPixels'
