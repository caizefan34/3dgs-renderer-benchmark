#!/bin/bash
# C19-2 Master Execution Script
# Runs all experiments in parallel across 8 A100 GPUs.
#
# Usage: bash scripts/c19-2/run_all_c19_2.sh
#
# GPU allocation:
#   GPU 0: canonical baseline repeat (Goal 0)
#   GPU 1: ncu resource/register measurement (Goal 1)
#   GPU 2: intersection replay 100-200 (Goal 2)
#   GPU 3: intersection replay 200-320 (Goal 2)
#   GPU 4: block-geometry control (Goal 3)
#   GPU 5: occupancy/stall characterization via ncu (Goal 4)
#   GPU 6: compiler/PTX/cubin resource inspection (via ptxas) (Goal 1)
#   GPU 7: independent repeat / validation (Goal 0)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

mkdir -p results/phase-c19 logs/c19-2

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NCU_SECTION_DIR="/usr/lib/nsight-compute/sections"
NCU_RULE_DIR="/usr/lib/nsight-compute/extras/RuleTemplates"

echo "=========================================="
echo "C19-2: Rasterizer Isolation & Resource-Pressure Gate"
echo "Started at $(date)"
echo "Working dir: $REPO_DIR"
echo "=========================================="

# =============================================================
# STEP 0: Fix ncu section path using a per-invocation workaround
# =============================================================
echo ""
echo "[Step 0] Verifying ncu section path workaround..."

# ncu needs --section-folder because the default path resolves to a missing dir
NCU_CMD="HOME=/tmp ncu --section-folder $NCU_SECTION_DIR --section-folder-recursive $NCU_RULE_DIR"

if $NCU_CMD --list-sections 2>&1 | grep -q "Occupancy"; then
    echo "  ncu sections OK"
else
    echo "  WARNING: ncu section path workaround may not work"
    echo "  Debug output:"
    $NCU_CMD --list-sections 2>&1 | head -5
fi

# =============================================================
# STEP 1: Fix ncu with symlink
# =============================================================
echo ""
echo "[Step 0b] Attempting to fix ncu symlink..."
# Check if the sections dir is already fixed
if [ -d "/usr/lib/x86_64-linux-gnu/nsight-compute/sections" ]; then
    echo "  Symlink already exists"
else
    echo "  Need to create symlink (may need password)"
    # Try without sudo first - if it fails, we'll use --section-folder approach
    ln -sf /usr/lib/nsight-compute/sections /usr/lib/x86_64-linux-gnu/nsight-compute/sections 2>/dev/null && \
        echo "  Symlink created" || \
        echo "  Cannot symlink (no sudo), will use --section-folder workaround"
fi

# =============================================================
# GOAL 0: Reproducibility Baseline (GPU 0, GPU 7)
# =============================================================
echo ""
echo "========== GOAL 0: Reproducibility Baseline =========="

echo "  [GPU 0] Reproducibility baseline..."
CUDA_VISIBLE_DEVICES=0 python3 scripts/c19-2/00_reproducibility_baseline.py \
    > logs/c19-2/goal0_gpu0_${TIMESTAMP}.log 2>&1 &
PID_GOAL0=$!
echo "    PID: $PID_GOAL0"

echo "  [GPU 7] Validation repeat..."
CUDA_VISIBLE_DEVICES=7 python3 scripts/c19-2/00_reproducibility_baseline.py \
    > logs/c19-2/goal0_gpu7_${TIMESTAMP}.log 2>&1 &
PID_GOAL0_VAL=$!
echo "    PID: $PID_GOAL0_VAL"

# =============================================================
# GOAL 1: ncu Resource/Register Measurement (GPU 1, GPU 6)
# =============================================================
echo ""
echo "========== GOAL 1: ncu Resource/Register Measurement =========="

# We'll profile tile=16 (below threshold) and tile=20 (above threshold)
for TS in 16 20; do
    echo "  [GPU 1] ncu profile tile_size=$TS..."
    CUDA_VISIBLE_DEVICES=1 \
    HOME=/tmp \
    ncu --section-folder $NCU_SECTION_DIR \
        --section-folder-recursive $NCU_RULE_DIR \
        --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
        --launch-skip 3 \
        --launch-count 1 \
        --set full \
        --csv \
        -o results/phase-c19/ncu_tile${TS} \
        python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size $TS --gpu 1 \
        > logs/c19-2/goal1_ncu_tile${TS}_${TIMESTAMP}.log 2>&1 &
    echo "    PID: $!"
done

# Also try ptxas verbose for resource info on GPU 6
echo "  [GPU 6] ptxas resource inspection..."
python3 -c "
import torch, gsplat, os
import subprocess
# Find the .so or force JIT compile with verbose ptxas
gsplat_dir = os.path.dirname(gsplat.__file__)
print(f'gsplat dir: {gsplat_dir}')
# Check for pre-compiled cubins
so_path = os.path.join(gsplat_dir, 'cuda', 'gsplat_cuda.cpython-310-x86_64-linux-gnu.so')
print(f'.so exists: {os.path.exists(so_path)}')
if os.path.exists(so_path):
    # Extract cubins
    result = subprocess.run(['cuobjdump', '-sass', so_path], 
                          capture_output=True, text=True, timeout=30)
    # Find rasterize kernel info
    for line in result.stdout.split(chr(10)):
        if 'rasterize_to_pixels_3dgs_fwd' in line:
            print(line)
    if 'registers' in result.stdout:
        for line in result.stdout.split(chr(10)):
            if 'registers' in line.lower() and 'rasterize' in line.lower():
                print(line)
" > logs/c19-2/goal6_ptxas_${TIMESTAMP}.log 2>&1 &
echo "    PID: $!"

# =============================================================
# GOAL 2: Rasterizer Replay (GPU 2, GPU 3)
# =============================================================
echo ""
echo "========== GOAL 2: Rasterizer Replay / Isolation =========="

echo "  [GPU 2] Replay ints 100-200..."
CUDA_VISIBLE_DEVICES=2 python3 scripts/c19-2/02_rasterizer_replay.py \
    > logs/c19-2/goal2_replay_gpu2_${TIMESTAMP}.log 2>&1 &
PID_REPLAY1=$!
echo "    PID: $PID_REPLAY1"

echo "  [GPU 3] Replay ints 200-320..."
# We'll run the replay script on both, it covers all target ints
CUDA_VISIBLE_DEVICES=3 python3 scripts/c19-2/02_rasterizer_replay.py \
    > logs/c19-2/goal2_replay_gpu3_${TIMESTAMP}.log 2>&1 &
PID_REPLAY2=$!
echo "    PID: $PID_REPLAY2"

# =============================================================
# GOAL 3: Block-Geometry Control (GPU 4)
# =============================================================
echo ""
echo "========== GOAL 3: Block-Geometry Control =========="

echo "  [GPU 4] Block geometry sweep..."
CUDA_VISIBLE_DEVICES=4 python3 scripts/c19-2/03_block_geometry_control.py \
    > logs/c19-2/goal3_block_geo_gpu4_${TIMESTAMP}.log 2>&1 &
PID_BLOCKGEO=$!
echo "    PID: $PID_BLOCKGEO"

# =============================================================
# GOAL 4: Mechanism Discrimination via ncu (GPU 5)
# =============================================================
echo ""
echo "========== GOAL 4: Occupancy/Stall Characterization =========="

echo "  [GPU 5] ncu warp stall metrics tile=16..."
CUDA_VISIBLE_DEVICES=5 \
HOME=/tmp \
ncu --section-folder $NCU_SECTION_DIR \
    --section-folder-recursive $NCU_RULE_DIR \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 \
    --launch-count 1 \
    --set full \
    --csv \
    -o results/phase-c19/ncu_stalls_tile16 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 16 --gpu 5 \
    > logs/c19-2/goal4_ncu_stalls_tile16_${TIMESTAMP}.log 2>&1 &
echo "    PID: $!"

echo "  [GPU 5] ncu warp stall metrics tile=20..."
CUDA_VISIBLE_DEVICES=5 \
HOME=/tmp \
ncu --section-folder $NCU_SECTION_DIR \
    --section-folder-recursive $NCU_RULE_DIR \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 \
    --launch-count 1 \
    --set full \
    --csv \
    -o results/phase-c19/ncu_stalls_tile20 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 20 --gpu 5 \
    > logs/c19-2/goal4_ncu_stalls_tile20_${TIMESTAMP}.log 2>&1 &
echo "    PID: $!"

# =============================================================
# WAIT FOR COMPLETION
# =============================================================
echo ""
echo "=========================================="
echo "All experiments launched. Waiting for completion..."
echo "=========================================="

FAILED=0

wait $PID_GOAL0 || { echo "  Goal 0 (GPU 0) FAILED"; FAILED=$((FAILED+1)); }
echo "  Goal 0 (GPU 0) complete"

wait $PID_GOAL0_VAL || { echo "  Goal 0 val (GPU 7) FAILED"; FAILED=$((FAILED+1)); }
echo "  Goal 0 val (GPU 7) complete"

wait $PID_REPLAY1 || { echo "  Goal 2 replay (GPU 2) FAILED"; FAILED=$((FAILED+1)); }
echo "  Goal 2 replay (GPU 2) complete"

wait $PID_REPLAY2 || { echo "  Goal 2 replay (GPU 3) FAILED"; FAILED=$((FAILED+1)); }
echo "  Goal 2 replay (GPU 3) complete"

wait $PID_BLOCKGEO || { echo "  Goal 3 block geo (GPU 4) FAILED"; FAILED=$((FAILED+1)); }
echo "  Goal 3 block geo (GPU 4) complete"

# Wait for background ncu jobs
wait
echo "  All ncu profiling jobs complete"

echo ""
echo "=========================================="
echo "All experiments finished. Failures: $FAILED"
echo "=========================================="

# Quick summary
echo ""
echo "=== OUTPUT FILES ==="
find results/phase-c19/ -name "c19-2_*" -o -name "ncu_tile*" 2>/dev/null | sort
echo ""
echo "=== LOG FILES ==="
ls -la logs/c19-2/

# Generate report if all data is available
echo ""
echo "Generating final report..."
python3 scripts/c19-2/99_generate_report.py 2>&1

echo ""
echo "DONE at $(date)"
