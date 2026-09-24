#!/bin/bash
# Fix ncu CUDA_VISIBLE_DEVICES issue and run profiling on GPU 0-1
# When CUDA_VISIBLE_DEVICES=1 is set, that GPU appears as cuda:0 inside the container
# So --gpu must be 0, not 1.

REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
NCU_SEC=/usr/lib/nsight-compute/sections
NCU_SEC_REC=/usr/lib/nsight-compute/extras/RuleTemplates

cd $REPO

echo "=== Running ncu profile tile=16 on GPU 1 (appears as cuda:0) ==="
CUDA_VISIBLE_DEVICES=1 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o $REPO/results/phase-c19/ncu_tile16 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 16 --gpu 0 \
    2>&1

# Try to find where ncu saved the output
find $REPO -name "ncu_tile16*" 2>/dev/null
find /tmp -name "ncu_tile16*" 2>/dev/null
