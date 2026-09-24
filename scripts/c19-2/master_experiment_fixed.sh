#!/bin/bash
# Fixed C19-2 master experiment script for mx (8×A100)
# PYTHONPATH fixed for ncu + CUDA_VISIBLE_DEVICES GPU numbering fix.
set -x
REPO_DIR=/home/liaoyuanjun/3dgs-renderer-benchmark
cd $REPO_DIR
mkdir -p logs/c19-2 results/phase-c19
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NCU_SEC=/usr/lib/nsight-compute/sections
NCU_SEC_REC=/usr/lib/nsight-compute/extras/RuleTemplates
NCU_PYTHON="HOME=/tmp PYTHONPATH=/home/liaoyuanjun/.local/lib/python3.10/site-packages"
NCU_BASE="$NCU_PYTHON ncu --section-folder $NCU_SEC --kernel-name rasterize_to_pixels_3dgs_fwd_kernel --launch-skip 3 --launch-count 1 --set full -c csv"
PROFILE_SCRIPT="python3 /home/liaoyuanjun/3dgs-renderer-benchmark/scripts/c19-2/01_ncu_resource_profile.py"

echo "[$(date)] GPU count: $(nvidia-smi -L | wc -l)"

# ===== GOAL 0: Reproducibility (GPU 0 & GPU 7) =====
echo "[$(date)] Goal 0: Reproducibility (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/c19-2/00_reproducibility_baseline.py > logs/c19-2/goal0_gpu0.log 2>&1
echo "[$(date)] Goal 0 GPU 0 done (exit=$?)"

echo "[$(date)] Goal 0: Validation (GPU 7)"
CUDA_VISIBLE_DEVICES=7 python3 scripts/c19-2/00_reproducibility_baseline.py > logs/c19-2/goal0_gpu7.log 2>&1
echo "[$(date)] Goal 0 GPU 7 done (exit=$?)"

# ===== GOAL 1: ncu Resource Profile (GPU 1, one at a time) =====
for TS in 16 20 8 32; do
  echo "[$(date)] Goal 1: ncu tile=${TS} (GPU 1)"
  CUDA_VISIBLE_DEVICES=1 eval "$NCU_BASE -o results/phase-c19/ncu_tile${TS} $PROFILE_SCRIPT --tile-size ${TS} --gpu 0" \
    > logs/c19-2/goal1_ncu_tile${TS}.log 2>&1
  echo "[$(date)] Goal 1 ncu tile=${TS} done (exit=$?)"
done

# ===== GOAL 2: Rasterizer Replay (GPU 2 & GPU 3) =====
echo "[$(date)] Goal 2: Replay (GPU 2)"
CUDA_VISIBLE_DEVICES=2 python3 scripts/c19-2/02_rasterizer_replay.py > logs/c19-2/goal2_replay_gpu2.log 2>&1
echo "[$(date)] Goal 2 replay done (exit=$?)"

echo "[$(date)] Goal 2: Replay validation (GPU 3)"
CUDA_VISIBLE_DEVICES=3 python3 scripts/c19-2/02_rasterizer_replay.py > logs/c19-2/goal2_replay_gpu3.log 2>&1
echo "[$(date)] Goal 2 replay val done (exit=$?)"

# ===== GOAL 3: Block-Geometry Control (GPU 4) =====
echo "[$(date)] Goal 3: Block geometry (GPU 4)"
CUDA_VISIBLE_DEVICES=4 python3 scripts/c19-2/03_block_geometry_control.py > logs/c19-2/goal3_block_geo.log 2>&1
echo "[$(date)] Goal 3 done (exit=$?)"

# ===== GOAL 4: ncu Stalls (GPU 5) =====
for TS in 16 20 24; do
  echo "[$(date)] Goal 4: ncu stalls tile=${TS} (GPU 5)"
  CUDA_VISIBLE_DEVICES=5 eval "$NCU_BASE -o results/phase-c19/ncu_stalls_tile${TS} $PROFILE_SCRIPT --tile-size ${TS} --gpu 0" \
    > logs/c19-2/goal4_ncu_tile${TS}.log 2>&1
  echo "[$(date)] Goal 4 ncu stalls tile=${TS} done (exit=$?)"
done

# ===== GOAL 6: ptxas inspection =====
echo "[$(date)] Goal 6: ptxas/cuobjdump"
python3 -c "
import os, subprocess
so = '/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat/cuda/gsplat_cuda.cpython-310-x86_64-linux-gnu.so'
print(f'so exists: {os.path.exists(so)}')
if os.path.exists(so):
    r = subprocess.run(['cuobjdump', '-sass', so], capture_output=True, text=True, timeout=60)
    lines = r.stdout.split(chr(10))
    for i, ln in enumerate(lines):
        if 'rasterize_to_pixels_3dgs_fwd' in ln:
            start = max(0, i-2)
            end = min(len(lines), i+25)
            for j in range(start, end):
                print(lines[j])
            print('---')
" > logs/c19-2/goal6_ptxas.log 2>&1
echo "[$(date)] Goal 6 done (exit=$?)"

# ===== Export ncu results to CSV =====
echo "[$(date)] Exporting ncu results..."
for f in results/phase-c19/ncu_*.ncu-rep; do
  [ -f "$f" ] || continue
  csv="${f%.ncu-rep}.csv"
  $NCU_PYTHON ncu --import "$f" -c csv > "$csv" 2>/dev/null && echo "  Exported $csv" || echo "  FAILED: $f"
done

# ===== Generate report =====
echo "[$(date)] Generating report..."
python3 scripts/c19-2/99_generate_report.py 2>&1
echo "[$(date)] Report done (exit=$?)"

# ===== Summary =====
echo ""
echo "=== RESULT FILES ==="
find results/phase-c19/ -maxdepth 1 -name "c19-2_*" -o -name "ncu_*" | sort
echo ""
echo "=== LOG FILES ==="
ls -la logs/c19-2/
echo "[$(date)] ALL DONE"
