#!/bin/bash
# Run ncu profiling on GPU 1
# CUDA_VISIBLE_DEVICES masks GPU numbering: GPU 1 appears as cuda:0 to the process
# So we use --gpu 0 in the script

REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
NCU_SEC=/usr/lib/nsight-compute/sections
NCU_SEC_REC=/usr/lib/nsight-compute/extras/RuleTemplates
cd $REPO

TILE=${1:-16}
echo "ncu profiling tile_size=$TILE on GPU 1"

CUDA_VISIBLE_DEVICES=1 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o results/phase-c19/ncu_tile${TILE} \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size $TILE --gpu 0 \
    2>&1

echo "Exit code: $?"
echo "Outputs:"
find results/phase-c19/ -name "ncu_tile${TILE}*" 2>/dev/null
