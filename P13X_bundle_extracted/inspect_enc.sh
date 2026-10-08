#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
echo "=== isect id encoding in csrc ==="
grep -rn "<< 32\|<<32\|isect_id\b" $WT/gsplat/csrc/sampling.cuh $WT/gsplat/csrc/*.cu 2>/dev/null | grep -i "isect\|tile\|depth" | head -15
echo
echo "=== intersect_tile kernel signature files ==="
grep -rln "intersect_tile" $WT/gsplat/csrc/ | head
echo
echo "=== encoding lines from the tile intersect impl ==="
grep -rn "tile_idx\|cam_idx" $WT/gsplat/csrc/intersection.cuh 2>/dev/null | head -10
grep -rn "int64_t.*isect\|isect.*int64" $WT/gsplat/csrc/*.cuh 2>/dev/null | head -10
