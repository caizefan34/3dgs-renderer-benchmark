#!/bin/bash
# Round-62 exp3: controlled in-wave A/B on garden 3000 steps seed 0.
#   GPU0: ctrl (R60 masked-Adam)   GPU1: decay 0.999   GPU2: decay 0.99
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR62exp3
mkdir -p $OUT

common="--base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 3000 --width 1280 --height 720 --seed 0 --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 --lr-decay 0.1 --densify-window 1500 --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --anchor-densify --anchor-densify-every 2 --cull-interval 1 --masked-adam"
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --out $OUT/ctrl.json > $OUT/ctrl.log 2>&1; echo CTRL_RC=$? > $OUT/ctrl.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --masked-adam-union-decay 0.999 --out $OUT/d0999.json > $OUT/d0999.log 2>&1; echo D999_RC=$? > $OUT/d0999.rc) &
(CUDA_VISIBLE_DEVICES=2 $PY $BENCH $common --masked-adam-union-decay 0.99 --out $OUT/d099.json > $OUT/d099.log 2>&1; echo D099_RC=$? > $OUT/d099.rc) &
wait
echo EXP3_DONE
cat $OUT/ctrl.rc $OUT/d0999.rc $OUT/d099.rc