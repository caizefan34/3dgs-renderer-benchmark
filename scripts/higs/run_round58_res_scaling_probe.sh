#!/bin/bash
# Round-58 mechanism probe (2026-08-04): per-step cost at 360p vs 720p.
# Verifies the R58 negative finding's mechanism: at a 720p target the
# progressive-res coarse stage (360p) has ~0-9% per-step saving because the
# step cost is dominated by resolution-invariant per-Gaussian stages
# (projection/SH/backward). 200 steps, eg r=0.35, same recipe as R58.
# Measured (A100): garden 360p 18.9ms vs 720p 20.8ms; train 360p 15.1ms vs
# 720p 12.2ms (not cheaper; per-Gaussian floor + warmup). See r58-summary.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR58probe
mkdir -p $OUT
run_p () {
  local gpu=$1 scene=$2 w=$3 h=$4 tag=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 200 --width $w --height $h --seed 0 \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 200 \
    --lr-decay 1.0 --densify-window 200 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}
run_p 0 tanks_and_temples/train 1280 720 p_train_720p &
run_p 1 tanks_and_temples/train 640 360 p_train_360p &
run_p 2 mipnerf360/garden 1280 720 p_garden_720p &
run_p 3 mipnerf360/garden 640 360 p_garden_360p &
wait
echo ALL_DONE_PROBE