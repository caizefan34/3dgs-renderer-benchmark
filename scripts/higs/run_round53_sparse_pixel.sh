#!/bin/bash
# Round 53 (M6 baseline 3/3): Speedy-Splat-style sparse-pixel training signal.
# --sampling-mode sparse_pixel --pixel-sampling-ratio 0.35: each step keeps
# 35% of pixels iid and takes the pixel-subsampled L1 mean (unbiased estimator
# of the full-frame mean, same argument as the tile-masked loss). The frozen
# gsplat HiGS rasterizer has no pixel-sparse path, so the render is full-frame:
# this arm reproduces Speedy-Splat's TRAINING SIGNAL (not its wall speed) to
# compare sampling-signal quality at matched pixel coverage vs the tile arms.
# Recipe identical to round-50/51/52 otherwise: full-res LPIPS every 25, high-N
# scenes + anchor densify every 2, train no anchor. 3 seeds per scene.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR53
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 anchor=$4 tag=$5
  local extra=""
  if [ "$anchor" = "1" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 1.0 --sampling-mode sparse_pixel --pixel-sampling-ratio 0.35 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# wave 1: train (no anchor) + garden (anchor)
run_dyn 0 tanks_and_temples/train 0 0 r53_train_px_s0 &
run_dyn 1 tanks_and_temples/train 1 0 r53_train_px_s1 &
run_dyn 2 tanks_and_temples/train 2 0 r53_train_px_s2 &
run_dyn 3 mipnerf360/garden 0 1 r53_garden_px_s0 &
run_dyn 4 mipnerf360/garden 1 1 r53_garden_px_s1 &
run_dyn 5 mipnerf360/garden 2 1 r53_garden_px_s2 &
wait
echo WAVE1_DONE
# wave 2: bicycle (anchor)
run_dyn 0 mipnerf360/bicycle 0 1 r53_bicycle_px_s0 &
run_dyn 1 mipnerf360/bicycle 1 1 r53_bicycle_px_s1 &
run_dyn 2 mipnerf360/bicycle 2 1 r53_bicycle_px_s2 &
wait
echo ALL_DONE_R53
