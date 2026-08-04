#!/bin/bash
# R51b rerun: train uniform r=0.30 seed 1 (suspected outlier, PSNR 16.07 vs 16.8-16.9)
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR51b
CUDA_VISIBLE_DEVICES=0 $PY $BENCH --base-dir $BASE --scene tanks_and_temples/train --backends higs_dynamic_ts \
  --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed 1 \
  --tile-sampling-ratio 0.30 --sampling-mode uniform --error-alpha 1.0 \
  --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
  --lr-decay 0.1 --densify-window 1500 \
  --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
  --out $OUT/r51b_train_uniform_r030_s1_re.json > $OUT/r51b_train_uniform_r030_s1_re.log 2>&1
echo ALL_DONE_R51B_RE
