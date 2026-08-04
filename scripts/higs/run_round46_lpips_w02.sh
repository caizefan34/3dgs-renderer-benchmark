#!/bin/bash
# Round 46: LPIPS-weight probe (w=0.2 vs w=0.1) on the high-N scenes.
#
# Rounds 43/44/45 closed the sampling-side and densify-side levers; the
# high-N honest bound is perceptual (LPIPS +0.050 garden/bicycle). LPIPS
# regularization (w=0.1 every 25 at full res) is already part of the recipe;
# doubling the weight to 0.2 tests whether the bound can be compressed from
# the loss side without hurting PSNR.
#
# Protocol: identical to round-41d/42 recommended op point
# (error_guided r=0.35 lambda=0.7 + --lpips-full-res, R36 recipe,
# densify-every 5, 3000 steps, 1920x1080, n-train 4 / n-eval 3), 3 seeds,
# only change: --lpips-loss-weight 0.2.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=${OUT:-/tmp/qsR46}
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.2 --lpips-loss-every 25 --lpips-full-res \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

run_garden () {
  run_dyn 0 mipnerf360/garden 0 r46_garden_eg035_l07_fr_w02_s0
  run_dyn 0 mipnerf360/garden 1 r46_garden_eg035_l07_fr_w02_s1
  run_dyn 0 mipnerf360/garden 2 r46_garden_eg035_l07_fr_w02_s2
}
run_bicycle () {
  run_dyn 1 mipnerf360/bicycle 0 r46_bicycle_eg035_l07_fr_w02_s0
  run_dyn 1 mipnerf360/bicycle 1 r46_bicycle_eg035_l07_fr_w02_s1
  run_dyn 1 mipnerf360/bicycle 2 r46_bicycle_eg035_l07_fr_w02_s2
}
run_garden &
run_bicycle &
wait
echo ALL_DONE_R46