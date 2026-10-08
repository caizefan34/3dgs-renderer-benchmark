#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
R=/root/3dgs-renderer-benchmark
echo "=== reference_v1 config ==="
grep -n "densify_until_iter\|densify_from_iter\|densification_interval\|opacity_reset_interval\|sh_progress_interval\|eval_iterations\|lambda_dssim\|sh_degree\|max_sh\|seed" $R/baseline/reference_v1/config.py | head -25
echo
echo "=== isect_offset_encode wrapper ==="
grep -n "def isect_offset_encode" -A 30 $WT/gsplat/cuda/_wrapper.py | head -45
echo
echo "=== isect_tiles wrapper (encoding doc) ==="
grep -n "def isect_tiles" -A 45 $WT/gsplat/cuda/_wrapper.py | head -60
echo
echo "=== GTDataset interface ==="
grep -n "class GTDataset\|def __len__\|def get_item\|def get_camera\|def __init__" $R/scripts/epic05/phase7/dataset.py 2>/dev/null | head
ls $R/baseline/reference_v1/ | head
echo
echo "=== p12 preflight tolerance ==="
grep -n "tol\|abs\|match\|bitwise\|assert" /root/epic05_p12/p12_preflight.py | head -20
