#!/bin/bash
# Round-62 exp1 smoke: union-invisible frozen-row opacity decay, garden 720p 300 steps.
#   GPU0: control (R60 k1 masked-Adam)   GPU1: decay 0.999   GPU2: decay 0.99
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR62exp1
mkdir -p $OUT

common="--base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 300 --width 1280 --height 720 --seed 0 --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 --error-refresh-every 25 --error-lambda 0.7 --eval-every 60 --lr-decay 0.1 --densify-window 1500 --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --anchor-densify --anchor-densify-every 2 --cull-interval 1 --masked-adam"
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --out $OUT/ctrl.json > $OUT/ctrl.log 2>&1; echo CTRL_RC=$? > $OUT/ctrl.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --masked-adam-union-decay 0.999 --out $OUT/decay0999.json > $OUT/decay0999.log 2>&1; echo D999_RC=$? > $OUT/decay0999.rc) &
(CUDA_VISIBLE_DEVICES=2 $PY $BENCH $common --masked-adam-union-decay 0.99 --out $OUT/decay099.json > $OUT/decay099.log 2>&1; echo D099_RC=$? > $OUT/decay099.rc) &
wait
echo EXP1_DONE
cat $OUT/ctrl.rc $OUT/decay0999.rc $OUT/decay099.rc