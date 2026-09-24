#!/bin/bash
# C19-2 Master v2 - fixes ncu eval expansion and replay script
set -x
REPO_DIR=/home/liaoyuanjun/3dgs-renderer-benchmark
cd $REPO_DIR

mkdir -p logs/c19-2 results/phase-c19
NCU_SEC=/usr/lib/nsight-compute/sections

echo "[$(date)] === C19-2 v2 Master ==="

# ===== GOAL 0: Already done, check reproducibility data exists =====
if [ ! -f results/phase-c19/c19-2_reproducibility.json ]; then
  echo "[$(date)] Goal 0: Reproducibility (GPU 0)"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/c19-2/00_reproducibility_baseline.py > logs/c19-2/goal0_v2_gpu0.log 2>&1
fi
echo "[$(date)] Reproducibility data: $(ls -la results/phase-c19/c19-2_reproducibility.json 2>/dev/null)"

# ===== GOAL 1: ncu - fixed direct command (no eval) =====
for TS in 16 20 8 32; do
  echo "[$(date)] ncu tile=${TS} (GPU 0)"
  CUDA_VISIBLE_DEVICES=0 \
  HOME=/tmp PYTHONPATH=/home/liaoyuanjun/.local/lib/python3.10/site-packages \
  ncu --section-folder $NCU_SEC \
      --kernel-name rasterize_to_pixels_3dgs_fwd_kernel \
      --launch-skip 3 \
      --launch-count 1 \
      --set full \
      -c csv \
      -o results/phase-c19/ncu_tile${TS} \
      python3 scripts/c19-2/01_ncu_resource_profile.py --tile-size $TS --gpu 0 \
      > logs/c19-2/goal1_v2_ncu_tile${TS}.log 2>&1
  echo "[$(date)] ncu tile=${TS} done (exit=$?)"
done

# ===== GOAL 2: Rasterizer Replay - fixed script =====
echo "[$(date)] Replay (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/c19-2/02_rasterizer_replay.py > logs/c19-2/goal2_v2_replay.log 2>&1
echo "[$(date)] Replay done (exit=$?)"

# ===== GOAL 3: Block-Geometry Control =====
echo "[$(date)] Block geometry (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/c19-2/03_block_geometry_control.py > logs/c19-2/goal3_v2_block_geo.log 2>&1
echo "[$(date)] Block geo done (exit=$?)"

# ===== Export ncu CSV =====
echo "[$(date)] Export ncu CSVs..."
for f in results/phase-c19/ncu_*.ncu-rep; do
  [ -f "$f" ] || continue
  HOME=/tmp PYTHONPATH=/home/liaoyuanjun/.local/lib/python3.10/site-packages \
  ncu --import "$f" -c csv > "${f%.ncu-rep}.csv" 2>/dev/null
  echo "  Exported: ${f%.ncu-rep}.csv"
done

echo "[$(date)] === ALL DONE ==="
echo ""
echo "=== FILES ==="
find results/phase-c19/ -maxdepth 1 \( -name "c19-2_*" -o -name "ncu_*" \) -ls 2>/dev/null
