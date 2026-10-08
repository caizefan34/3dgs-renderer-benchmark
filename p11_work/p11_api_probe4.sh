#!/bin/bash
set -u
T=/root/epic05_p8/gsplat-main-512d366b
echo "=== find autograd Function classes ==="
grep -rn 'autograd.Function' $T/gsplat/ --include='*.py' | grep -v __pycache__ | head
echo "=== rasterization decorator ==="
sed -n '225,240p' $T/gsplat/rendering.py
echo "=== how rasterization_3dgs is called ==="
grep -n 'rasterization_3dgs' $T/gsplat/rendering.py | head
echo "=== _Rasterization3DGS class? ==="
grep -rn 'class _' $T/gsplat/rendering.py | head
echo "=== bwd in Rendering.cpp ==="
grep -n 'rasterization_3dgs' $T/gsplat/cuda/Rasterization.cpp 2>/dev/null | head
ls $T/gsplat/cuda/*.cpp 2>/dev/null
echo "=== 3DGS fwd/bwd kernel files ==="
ls $T/gsplat/cuda/csrc/ | grep -iE '3dgs|isect|projection' | head -20
