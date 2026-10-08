#!/bin/bash
set -u
T=/root/epic05_p8/gsplat-main-512d366b
echo "=== version ==="
cat $T/gsplat/version.py
echo "=== num_contributing_gaussians ==="
sed -n '1668,1710p' $T/gsplat/cuda/_wrapper.py
echo "=== contributing_gaussian_ids ==="
sed -n '1776,1830p' $T/gsplat/cuda/_wrapper.py
echo "=== top_contributing ==="
sed -n '1895,1945p' $T/gsplat/cuda/_wrapper.py
echo "=== rendering.py head of rasterization ==="
sed -n '234,250p' $T/gsplat/rendering.py
echo "=== autograd func in rendering.py ==="
grep -n 'autograd\|_RasterizeToPixels\|class.*Function\|\.apply(' $T/gsplat/rendering.py | head -20
echo "=== backward call region ==="
sed -n '560,700p' $T/gsplat/rendering.py
