#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
GI=$WT/gsplat/experimental/render/functional/gaussian_inference.py
echo "=== _HigsAutogradFunction.forward native path (1830-2035) ==="
sed -n '1830,2035p' $GI
echo
echo "=== last_forward_metadata definition ==="
grep -n "last_forward_metadata" $GI | head -10
