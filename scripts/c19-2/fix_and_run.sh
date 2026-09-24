#!/bin/bash
# Fix and run approach: run ncu and experiments directly with correct GPU mapping

REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
cd $REPO

echo "1. Run ncu tile=16 on GPU 1 (maps to cuda:0)"
CUDA_VISIBLE_DEVICES=1 HOME=/tmp ncu \
    --section-folder /usr/lib/nsight-compute/sections \
    --section-folder-recursive /usr/lib/nsight-compute/extras/RuleTemplates \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o results/phase-c19/ncu_tile16 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 16 --gpu 0 \
    2>&1
echo "exit: $?"
ls -la results/phase-c19/ncu_tile16* 2>/dev/null || echo "ncu tile16 output not found"

echo ""
echo "2. Run ncu tile=20 on GPU 1"
CUDA_VISIBLE_DEVICES=1 HOME=/tmp ncu \
    --section-folder /usr/lib/nsight-compute/sections \
    --section-folder-recursive /usr/lib/nsight-compute/extras/RuleTemplates \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o results/phase-c19/ncu_tile20 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 20 --gpu 0 \
    2>&1
echo "exit: $?"
ls -la results/phase-c19/ncu_tile20* 2>/dev/null || echo "ncu tile20 output not found"

echo ""
echo "3. Run ncu tile=32 on GPU 2"
CUDA_VISIBLE_DEVICES=2 HOME=/tmp ncu \
    --section-folder /usr/lib/nsight-compute/sections \
    --section-folder-recursive /usr/lib/nsight-compute/extras/RuleTemplates \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o results/phase-c19/ncu_tile32 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 32 --gpu 0 \
    2>&1
echo "exit: $?"
ls -la results/phase-c19/ncu_tile32* 2>/dev/null || echo "ncu tile32 output not found"

echo ""
echo "4. Run ncu tile=8 on GPU 2"
CUDA_VISIBLE_DEVICES=2 HOME=/tmp ncu \
    --section-folder /usr/lib/nsight-compute/sections \
    --section-folder-recursive /usr/lib/nsight-compute/extras/RuleTemplates \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o results/phase-c19/ncu_tile8 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 8 --gpu 0 \
    2>&1
echo "exit: $?"
ls -la results/phase-c19/ncu_tile8* 2>/dev/null || echo "ncu tile8 output not found"

echo ""
echo "=== ALL OUTPUT FILES ==="
find results/phase-c19/ -name "ncu_*" 2>/dev/null | sort
