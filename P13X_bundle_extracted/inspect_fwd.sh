#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
GI=$WT/gsplat/experimental/render/functional/gaussian_inference.py
echo "=== _higs_dynamic_forward def + core call chain ==="
grep -n "_higs_dynamic_forward\|def _higs\|ext\.\(higs\)\?[a-z_]*forward\|rasterize_to_pixels\|projection\|intersect\|sort\|tile" $GI | grep -v "^.*#" | head -60
echo
echo "=== forward fn body sample (search higs_dynamic_forward def) ==="
N=$(grep -n "def _higs_dynamic_forward" $GI | cut -d: -f1)
echo "def at line $N"
sed -n "${N},$((N+90))p" $GI
