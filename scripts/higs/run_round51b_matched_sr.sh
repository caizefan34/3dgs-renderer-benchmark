#!/bin/bash
# Round 51b (M6 matched-sr arm): uniform random-tile at nominal r=0.30 so the
# realized sampling ratio matches error_guided r=0.35 (uniform realizes ~1.15-1.2x
# the nominal ratio: garden/bicycle sr 0.402 at r=0.35, train 0.376; eg sr 0.344-0.348).
# Target sr ~0.345 (garden/bicycle) / ~0.31 (train). 3 seeds each.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR51b
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 ratio=$4 anchor=$5 tag=$6
  local extra=""
  if [ "$anchor" = "1" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio $ratio --sampling-mode uniform --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# wave 1: train (no anchor) + garden (anchor)
run_dyn 0 tanks_and_temples/train 0 0.30 0 r51b_train_uniform_r030_s0 &
run_dyn 1 tanks_and_temples/train 1 0.30 0 r51b_train_uniform_r030_s1 &
run_dyn 2 tanks_and_temples/train 2 0.30 0 r51b_train_uniform_r030_s2 &
run_dyn 3 mipnerf360/garden 0 0.30 1 r51b_garden_uniform_r030_s0 &
run_dyn 4 mipnerf360/garden 1 0.30 1 r51b_garden_uniform_r030_s1 &
run_dyn 5 mipnerf360/garden 2 0.30 1 r51b_garden_uniform_r030_s2 &
wait
echo WAVE1_DONE
# wave 2: bicycle (anchor)
run_dyn 0 mipnerf360/bicycle 0 0.30 1 r51b_bicycle_uniform_r030_s0 &
run_dyn 1 mipnerf360/bicycle 1 0.30 1 r51b_bicycle_uniform_r030_s1 &
run_dyn 2 mipnerf360/bicycle 2 0.30 1 r51b_bicycle_uniform_r030_s2 &
wait
echo ALL_DONE_R51B
