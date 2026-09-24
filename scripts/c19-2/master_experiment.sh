#!/bin/bash
# Master experiment script for C19-2 on mx (8×A100)
# Run via: nohup bash scripts/c19-2/master_experiment.sh > logs/c19-2/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -x
REPO_DIR=/home/liaoyuanjun/3dgs-renderer-benchmark
cd $REPO_DIR

mkdir -p logs/c19-2 results/phase-c19

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NCU_SEC=/usr/lib/nsight-compute/sections
NCU_SEC_REC=/usr/lib/nsight-compute/extras/RuleTemplates

echo "[$(date)] Starting C19-2 master experiments"
echo "PWD: $PWD"
echo "GPU count: $(nvidia-smi -L | wc -l)"

# =============================================================
# GOAL 0: Reproducibility (GPU 0, GPU 7)
# =============================================================
echo "[$(date)] Goal 0: Reproducibility Baseline (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/c19-2/00_reproducibility_baseline.py \
    > logs/c19-2/goal0_gpu0.log 2>&1
echo "[$(date)] Goal 0: GPU 0 complete (exit: $?)"

echo "[$(date)] Goal 0: Validation (GPU 7)"
CUDA_VISIBLE_DEVICES=7 python3 scripts/c19-2/00_reproducibility_baseline.py \
    > logs/c19-2/goal0_gpu7.log 2>&1
echo "[$(date)] Goal 0: GPU 7 complete (exit: $?)"

# =============================================================
# GOAL 1: ncu Resource/Register Measurement (GPU 1)
# =============================================================
echo "[$(date)] Goal 1: ncu profile tile=16 (GPU 1)"
CUDA_VISIBLE_DEVICES=1 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o $REPO_DIR/results/phase-c19/ncu_tile16 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 16 --gpu 1 \
    > logs/c19-2/goal1_ncu_tile16.log 2>&1
echo "[$(date)] Goal 1: ncu tile=16 complete (exit: $?)"

echo "[$(date)] Goal 1: ncu profile tile=20 (GPU 1)"
CUDA_VISIBLE_DEVICES=1 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o $REPO_DIR/results/phase-c19/ncu_tile20 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 20 --gpu 1 \
    > logs/c19-2/goal1_ncu_tile20.log 2>&1
echo "[$(date)] Goal 1: ncu tile=20 complete (exit: $?)"

# Also profile tile=8 and tile=32 for comparison range
echo "[$(date)] Goal 1: ncu profile tile=8 (GPU 1)"
CUDA_VISIBLE_DEVICES=1 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o $REPO_DIR/results/phase-c19/ncu_tile8 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 8 --gpu 1 \
    > logs/c19-2/goal1_ncu_tile8.log 2>&1
echo "[$(date)] Goal 1: ncu tile=8 complete (exit: $?)"

echo "[$(date)] Goal 1: ncu profile tile=32 (GPU 1)"
CUDA_VISIBLE_DEVICES=1 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o $REPO_DIR/results/phase-c19/ncu_tile32 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 32 --gpu 1 \
    > logs/c19-2/goal1_ncu_tile32.log 2>&1
echo "[$(date)] Goal 1: ncu tile=32 complete (exit: $?)"

# =============================================================
# GOAL 2: Rasterizer Replay (GPU 2, GPU 3)
# =============================================================
echo "[$(date)] Goal 2: Rasterizer Replay (GPU 2)"
CUDA_VISIBLE_DEVICES=2 python3 scripts/c19-2/02_rasterizer_replay.py \
    > logs/c19-2/goal2_replay_gpu2.log 2>&1
echo "[$(date)] Goal 2: Replay complete (exit: $?)"

echo "[$(date)] Goal 2: Rasterizer Replay validation (GPU 3)"
CUDA_VISIBLE_DEVICES=3 python3 scripts/c19-2/02_rasterizer_replay.py \
    > logs/c19-2/goal2_replay_gpu3.log 2>&1
echo "[$(date)] Goal 2: Replay val complete (exit: $?)"

# =============================================================
# GOAL 3: Block-Geometry Control (GPU 4)
# =============================================================
echo "[$(date)] Goal 3: Block-Geometry Control (GPU 4)"
CUDA_VISIBLE_DEVICES=4 python3 scripts/c19-2/03_block_geometry_control.py \
    > logs/c19-2/goal3_block_geo_gpu4.log 2>&1
echo "[$(date)] Goal 3: Block geometry complete (exit: $?)"

python3 scripts/c19-2/03_block_geometry_control.py

# =============================================================
# GOAL 4: Occupancy/Stall Characterization via ncu (GPU 5)
# =============================================================
echo "[$(date)] Goal 4: ncu stalls tile=16 (GPU 5)"
CUDA_VISIBLE_DEVICES=5 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o $REPO_DIR/results/phase-c19/ncu_stalls_tile16 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 16 --gpu 5 \
    > logs/c19-2/goal4_ncu_stalls_tile16.log 2>&1
echo "[$(date)] Goal 4: ncu stalls tile=16 complete (exit: $?)"

echo "[$(date)] Goal 4: ncu stalls tile=20 (GPU 5)"
CUDA_VISIBLE_DEVICES=5 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o $REPO_DIR/results/phase-c19/ncu_stalls_tile20 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 20 --gpu 5 \
    > logs/c19-2/goal4_ncu_stalls_tile20.log 2>&1
echo "[$(date)] Goal 4: ncu stalls tile=20 complete (exit: $?)"

echo "[$(date)] Goal 4: ncu stalls tile=24 (GPU 5)"
CUDA_VISIBLE_DEVICES=5 HOME=/tmp ncu \
    --section-folder $NCU_SEC \
    --section-folder-recursive $NCU_SEC_REC \
    --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
    --launch-skip 3 --launch-count 1 \
    --set full \
    --csv \
    -o $REPO_DIR/results/phase-c19/ncu_stalls_tile24 \
    python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size 24 --gpu 5 \
    > logs/c19-2/goal4_ncu_stalls_tile24.log 2>&1
echo "[$(date)] Goal 4: ncu stalls tile=24 complete (exit: $?)"

# =============================================================
# GOAL 6: ptxas resource inspection (GPU 6)
# =============================================================
echo "[$(date)] Goal 6: ptxas resource inspection via cuobjdump"
python3 -c "
import torch, gsplat, os, subprocess
gsplat_dir = os.path.dirname(gsplat.__file__)
so_path = os.path.join(gsplat_dir, 'cuda', 'gsplat_cuda.cpython-310-x86_64-linux-gnu.so')
print(f'so exists: {os.path.exists(so_path)}')
if os.path.exists(so_path):
    r = subprocess.run(['cuobjdump', '-sass', so_path], capture_output=True, text=True, timeout=60)
    lines = r.stdout.split(chr(10))
    print(f'Total lines: {len(lines)}')
    # Find rasterize kernel sections
    for i, ln in enumerate(lines):
        if 'rasterize_to_pixels_3dgs_fwd' in ln:
            # Print context
            start = max(0, i-2)
            end = min(len(lines), i+20)
            print('--- Context ---')
            for j in range(start, end):
                print(lines[j])
            print('---')
" > logs/c19-2/goal6_ptxas_${TIMESTAMP}.log 2>&1
echo "[$(date)] Goal 6: ptxas complete (exit: $?)"

# =============================================================
# GENERATE REPORT
# =============================================================
echo "[$(date)] Generating report..."
python3 scripts/c19-2/99_generate_report.py 2>&1
echo "[$(date)] Report generation complete (exit: $?)"

# =============================================================
# COLLECT ALL ncu outputs (convert .ncu-rep to readable CSV if needed)
# =============================================================
echo "[$(date)] Converting ncu outputs..."
# ncu saves .ncu-rep files, need to re-export
for f in results/phase-c19/ncu_tile*.ncu-rep results/phase-c19/ncu_stalls_*.ncu-rep; do
    if [ -f "$f" ]; then
        echo "  Found: $f"
        csvname="${f%.ncu-rep}.csv"
        HOME=/tmp ncu --section-folder $NCU_SEC --import "$f" --csv > "$csvname" 2>/dev/null
        echo "    Exported: $csvname"
    fi
done

# List all outputs
echo ""
echo "=== OUTPUT FILES ==="
find results/phase-c19/ -name "c19-2_*" -o -name "ncu_*" 2>/dev/null | sort

echo ""
echo "=== LOG FILES ==="
ls -la logs/c19-2/

echo "[$(date)] ALL DONE"
