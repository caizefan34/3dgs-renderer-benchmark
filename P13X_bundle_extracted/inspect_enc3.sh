#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
echo "=== IntersectTile.cu lines 165-210 (encoding) ==="
sed -n '165,210p' $WT/gsplat/cuda/csrc/IntersectTile.cu
echo
echo "=== intersect_offset_kernel lines 925-990 ==="
sed -n '925,990p' $WT/gsplat/cuda/csrc/IntersectTile.cu
