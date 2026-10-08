#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
GI=$WT/gsplat/experimental/render/functional/gaussian_inference.py
echo "=== captured tuple (2260-2340) ==="
sed -n '2260,2340p' $GI
echo
echo "=== culling in _HigsAutogradFunction.forward: where visible_ids computed (1700-1830) ==="
sed -n '1704,1830p' $GI
