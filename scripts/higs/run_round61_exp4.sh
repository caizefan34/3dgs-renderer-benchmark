#!/bin/bash
# Round-61 exp4: 300-step garden smoke to measure full-run tradeoff.
#   Wave1: control (masked-adam only) + A (min_frozen 10, refresh 0)
#   Wave2: B (min_frozen 10, refresh 2) + C (min_frozen 12, refresh 0)
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR61exp4
mkdir -p $OUT

common="--base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 300 --width 1280 --height 720 --seed 0 --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 --error-refresh-every 25 --error-lambda 0.7 --eval-every 60 --lr-decay 0.1 --densify-window 1500 --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --anchor-densify --anchor-densify-every 2 --cull-interval 1 --masked-adam"
# Wave 1
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --out $OUT/exp_ctrl300.json > $OUT/exp_ctrl300.log 2>&1; echo CTRL_RC=$? > $OUT/ctrl.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --mask-prune --mask-prune-min-frozen 10 --mask-prune-eval-refresh 0 --out $OUT/exp_A.json > $OUT/exp_A.log 2>&1; echo A_RC=$? > $OUT/A.rc) &
wait
echo WAVE1_DONE
# Wave 2
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --mask-prune --mask-prune-min-frozen 10 --mask-prune-eval-refresh 2 --out $OUT/exp_B.json > $OUT/exp_B.log 2>&1; echo B_RC=$? > $OUT/B.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --mask-prune --mask-prune-min-frozen 12 --mask-prune-eval-refresh 0 --out $OUT/exp_C.json > $OUT/exp_C.log 2>&1; echo C_RC=$? > $OUT/C.rc) &
wait
echo EXP4_DONE
cat $OUT/ctrl.rc $OUT/A.rc $OUT/B.rc $OUT/C.rc