#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
echo "=== isect id encoding in Intersect.h / IntersectTile.cu ==="
grep -n "isect_id\|<< 32\|<<32\|tile_id\|depth" $WT/gsplat/cuda/csrc/Intersect.h | head -25
echo
grep -n "isect_id\|<< 32\|<<32\|tile_id" $WT/gsplat/cuda/csrc/IntersectTile.cu | head -20
echo
echo "=== intersect_offset op (offsets convention) ==="
grep -rn "intersect_offset\|offset_encode" $WT/gsplat/cuda/csrc/*.cu $WT/gsplat/cuda/csrc/*.h 2>/dev/null | head -10
