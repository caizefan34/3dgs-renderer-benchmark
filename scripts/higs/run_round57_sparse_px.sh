#!/bin/bash
# Round 57 (M6 收尾): renderer-level sparse-pixel rasterization
# (higs_sparse_px). Speedy-Splat-style pixel-sparse rendering at the renderer
# level: upstream gsplat sparse kernels rasterize ONLY the pixels drawn by the
# per-step iid Bernoulli mask (packed output), so the renderer touches only
# active tiles/pixels. Arm A = --pixel-raster-ratio 0.35; arm B = 1.0 (its own
# dense baseline, identical composed pipeline). Quality/speed measured within
# the same backend so the pixel-fraction lever is isolated.
# Recipe identical to round 50/51/52/53: full-res LPIPS every 25, lr-decay,
# densify-window 1500, high-N scenes + anchor densify every 2, train no anchor.
# 3 seeds per scene x 2 arms = 18 runs.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR57
mkdir -p $OUT

run_px () {
  local gpu=$1 scene=$2 seed=$3 anchor=$4 ratio=$5 tag=$6
  local extra=""
  if [ "$anchor" = "1" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_sparse_px \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --pixel-raster-ratio $ratio \
    --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# wave 1: train (no anchor) x 3 seeds x 2 arms + garden anchor seed0 (6 GPUs)
run_px 0 tanks_and_temples/train 0 0 0.35 r57_train_px035_s0 &
run_px 1 tanks_and_temples/train 0 0 1.0  r57_train_px100_s0 &
run_px 2 tanks_and_temples/train 1 0 0.35 r57_train_px035_s1 &
run_px 3 tanks_and_temples/train 1 0 1.0  r57_train_px100_s1 &
run_px 4 tanks_and_temples/train 2 0 0.35 r57_train_px035_s2 &
run_px 5 tanks_and_temples/train 2 0 1.0  r57_train_px100_s2 &
wait
echo WAVE1_DONE
# wave 2: garden (anchor) x 3 seeds x 2 arms
run_px 0 mipnerf360/garden 0 1 0.35 r57_garden_px035_s0 &
run_px 1 mipnerf360/garden 0 1 1.0  r57_garden_px100_s0 &
run_px 2 mipnerf360/garden 1 1 0.35 r57_garden_px035_s1 &
run_px 3 mipnerf360/garden 1 1 1.0  r57_garden_px100_s1 &
run_px 4 mipnerf360/garden 2 1 0.35 r57_garden_px035_s2 &
run_px 5 mipnerf360/garden 2 1 1.0  r57_garden_px100_s2 &
wait
echo WAVE2_DONE
# wave 3: bicycle (anchor) x 3 seeds x 2 arms
run_px 0 mipnerf360/bicycle 0 1 0.35 r57_bicycle_px035_s0 &
run_px 1 mipnerf360/bicycle 0 1 1.0  r57_bicycle_px100_s0 &
run_px 2 mipnerf360/bicycle 1 1 0.35 r57_bicycle_px035_s1 &
run_px 3 mipnerf360/bicycle 1 1 1.0  r57_bicycle_px100_s1 &
run_px 4 mipnerf360/bicycle 2 1 0.35 r57_bicycle_px035_s2 &
run_px 5 mipnerf360/bicycle 2 1 1.0  r57_bicycle_px100_s2 &
wait
echo ALL_DONE_R57
