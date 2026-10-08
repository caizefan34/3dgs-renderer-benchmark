#!/bin/bash
# P11 gsplat API probe
set -u
T=/root/epic05_p8/gsplat-main-512d366b
echo "=== meta dict assembly in rendering.py ==="
grep -n 'meta = dict\|meta\[' $T/gsplat/rendering.py | head -20
echo "=== rasterize_to_pixels signature ==="
grep -n 'def rasterize_to_pixels' $T/gsplat/cuda/_wrapper.py | head
sed -n "$(grep -n 'def rasterize_to_pixels' $T/gsplat/cuda/_wrapper.py | head -1 | cut -d: -f1),+40p" $T/gsplat/cuda/_wrapper.py
echo "=== last_ids / contrib utilities ==="
grep -rn 'last_ids' $T/gsplat/cuda/_wrapper.py | head -10
grep -rn 'def .*contrib\|contributions' $T/gsplat/cuda/_wrapper.py | head -10
echo "=== rasterization return + backward path ==="
grep -n 'class _RasterizeToPixels\|def forward\|def backward' $T/gsplat/rendering.py | head -20
echo "=== visible pixels / tile count utils ==="
ls $T/gsplat/utils/
grep -rn 'def ' $T/gsplat/utils/compiled.py 2>/dev/null | head -30
