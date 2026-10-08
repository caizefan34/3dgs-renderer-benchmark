#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
GI=$WT/gsplat/experimental/render/functional/gaussian_inference.py
echo "=== _native_forward_capture body (2035-2260) ==="
sed -n '2035,2260p' $GI
