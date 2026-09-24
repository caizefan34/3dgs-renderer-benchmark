#!/bin/bash
# Run ncu profiling for all tile sizes with correct flags
# The key fix: no HOME=/tmp (breaks JIT cache), use --target-processes all

set -e
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
RES=$REPO/results/phase-c19
NCU_SEC=/usr/lib/nsight-compute/sections

# Fix ncu HOME issue
mkdir -p "/home/liaoyuanjun/Documents/NVIDIA Nsight Compute/2021.3.1/Sections" 2>/dev/null

cd $REPO

for TS in 16 20 8 32; do
    echo "=========================================="
    echo "ncu tile_size=$TS at $(date)"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 \
    ncu \
        --section-folder $NCU_SEC \
        --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
        --launch-skip 3 \
        --launch-count 1 \
        --set full \
        --csv \
        --target-processes all \
        -o $RES/ncu_tile${TS} \
        python3 $REPO/scripts/c19-2/01_ncu_resource_profile.py \
            --tile-size $TS \
            --gpu 0 \
        2>&1 | tail -20
    
    echo "Output files:"
    ls -la $RES/ncu_tile${TS}* 2>/dev/null
    echo ""
done

echo "=== All ncu profiles complete ==="
