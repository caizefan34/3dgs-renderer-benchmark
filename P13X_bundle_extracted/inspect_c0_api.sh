#!/bin/bash
# Inspect C0 renderer API surface for P13-X X2 instrumentation planning
WT=/root/epic05_frozen/higs_c0_final30k_worktree
echo "=== rasterize_gaussian_higs_dynamic signature ==="
grep -rn "def rasterize_gaussian_higs_dynamic" $WT/gsplat/ | head -3
echo
echo "=== experimental __init__ exports ==="
sed -n '1,40p' $WT/gsplat/experimental/__init__.py 2>/dev/null
echo
echo "=== backward_capture.py head ==="
head -60 $WT/backward_capture.py 2>/dev/null
echo
echo "=== gaussian_inference.py: return dict keys ==="
grep -n "densification_info\|def rasterize_gaussian_higs_dynamic\|tile_ind\|gaussian_ids\|camera_ids\|return " $WT/gsplat/experimental/render/functional/gaussian_inference.py 2>/dev/null | head -40
