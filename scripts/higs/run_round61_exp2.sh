#!/bin/bash
# Round-61 exp2: densify-end eval-mask refresh (fresh masks, no pending-backward
# conflict) on garden 720p 120-step smoke.
#   GPU0: --mask-prune --mask-prune-eval-refresh 1 (fresh eval mask, no gate)
#   GPU1: --mask-prune --mask-prune-eval-refresh 1 --mask-prune-opacity 0.1
# Controls: r60 masked-adam-only PSNR 22.49; r61 naive 20.66; opac-only 21.23.
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR61exp2
mkdir -p $OUT

common="--base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 120 --width 1280 --height 720 --seed 0 --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 --error-refresh-every 25 --error-lambda 0.7 --eval-every 60 --lr-decay 0.1 --densify-window 1500 --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --anchor-densify --anchor-densify-every 2 --cull-interval 1 --masked-adam --mask-prune --mask-prune-eval-refresh 1"
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --out $OUT/exp_fresh2.json > $OUT/exp_fresh2.log 2>&1; echo FRESH2_RC=$? > $OUT/fresh2.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --mask-prune-opacity 0.1 --out $OUT/exp_both.json > $OUT/exp_both.log 2>&1; echo BOTH_RC=$? > $OUT/both.rc) &
wait
echo EXP2_DONE
cat $OUT/fresh2.rc $OUT/both.rc