#!/bin/bash
# Round-60 smoke: garden 720p 120-step masked-adam vs control.
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR60smoke
mkdir -p $OUT

common="--base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 120 --width 1280 --height 720 --seed 0 --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 --error-refresh-every 25 --error-lambda 0.7 --eval-every 60 --lr-decay 0.1 --densify-window 1500 --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --anchor-densify --anchor-densify-every 2 --cull-interval 1"
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --masked-adam --out $OUT/smoke_ma.json > $OUT/smoke_ma.log 2>&1; echo MA_RC=$? > $OUT/ma.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --out $OUT/smoke_ctl.json > $OUT/smoke_ctl.log 2>&1; echo CTL_RC=$? > $OUT/ctl.rc) &
wait
echo SMOKE_DONE
cat $OUT/ma.rc $OUT/ctl.rc
