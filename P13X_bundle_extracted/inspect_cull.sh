#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
GI=$WT/gsplat/experimental/render/functional/gaussian_inference.py
echo "=== _cull_gaussians_batched signature ==="
N=$(grep -n "def _cull_gaussians_batched" $GI | cut -d: -f1)
sed -n "${N},$((N+30))p" $GI
echo
echo "=== _higs_dynamic_forward return dict keys ==="
N2=$(grep -n "def _higs_dynamic_forward" $GI | cut -d: -f1)
sed -n "$((N2+95)),$((N2+160))p" $GI | grep -n "frame\|alpha\|densification\|return\|\"" | head -20
