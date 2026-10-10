#!/bin/bash
set +e
SB=/mnt/storage_pool/liaoyuanjun/strong_baselines
echo "=== fastgs git commit (re-confirm) ==="
git -C $SB/fastgs rev-parse HEAD 2>/dev/null
git -C $SB/fastgs log -1 --format="%H %ci %s" 2>/dev/null | head -1
echo "=== fastgs train.py: argparse defaults for prune/budget control ==="
grep -nE "add_argument|densify_from_iter|densification_interval|opacity_reset|max_steps|iterations|prune|min_opacity|percent_dense|final_prune" $SB/fastgs/train.py 2>/dev/null | head -40
echo
echo "=== speedy-splat git commit (re-confirm) ==="
git -C $SB/speedy-splat rev-parse HEAD 2>/dev/null
echo "=== speedy-splat prune/budget knobs ==="
grep -rnE "add_argument|opacity|prune|densif|budget|max_steps|cap" $SB/speedy-splat/train.py 2>/dev/null | head -30
echo
echo "=== nerficg commit (gsplat-based, torch2.7 lineage) ==="
git -C /home/liaoyuanjun/3dgs-renderer-benchmark rev-parse HEAD 2>/dev/null
echo "DONE"
