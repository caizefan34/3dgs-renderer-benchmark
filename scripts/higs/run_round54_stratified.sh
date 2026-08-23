#!/bin/bash
# Round 54: stratified tile sampling (de-correlation lever from round-53).
# Round 53 (sparse-pixel signal) showed the high-N tile-sampling quality loss is
# sampling-correlation noise, not pixel count. The in-harness de-correlation
# lever is the rasterizer's "stratified" tile sampling (one tile per
# round(1/r)-tile stratum, spreading the draw across the frame each step).
# Recipe identical to round-50/51 otherwise: r=0.35, full-res LPIPS every 25,
# high-N scenes + anchor densify every 2, train no anchor. 3 seeds per scene.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR54
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 anchor=$4 tag=$5
  local extra=""
  if [ "$anchor" = "1" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode stratified \
    --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# wave 1: train (no anchor) + garden (anchor)
run_dyn 0 tanks_and_temples/train 0 0 r54_train_strat_s0 &
run_dyn 1 tanks_and_temples/train 1 0 r54_train_strat_s1 &
run_dyn 2 tanks_and_temples/train 2 0 r54_train_strat_s2 &
run_dyn 3 mipnerf360/garden 0 1 r54_garden_strat_s0 &
run_dyn 4 mipnerf360/garden 1 1 r54_garden_strat_s1 &
run_dyn 5 mipnerf360/garden 2 1 r54_garden_strat_s2 &
wait
echo WAVE1_DONE
# wave 2: bicycle (anchor)
run_dyn 0 mipnerf360/bicycle 0 1 r54_bicycle_strat_s0 &
run_dyn 1 mipnerf360/bicycle 1 1 r54_bicycle_strat_s1 &
run_dyn 2 mipnerf360/bicycle 2 1 r54_bicycle_strat_s2 &
wait
echo ALL_DONE_R54
