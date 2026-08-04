#!/bin/bash
# Round 56 (M5 extension): multi-resolution matrix.
# Recommended op point (error_guided r=0.35 lambda=0.7 + full-res LPIPS +
# high-N anchor every2, train no anchor) vs full at 540p and 720p on
# train/garden/bicycle, 3 seeds each. 1080p cells exist (rounds 41d/42/50).
# 3 scenes x 2 resolutions x 2 arms x 3 seeds = 36 runs.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR56
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 width=$4 height=$5 arm=$6 tag=$7
  local extra=""
  local ratio=1.0
  if [ "$arm" = "eg" ]; then
    ratio=0.35
    if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width $width --height $height --seed $seed \
    --tile-sampling-ratio $ratio --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# wave 1: 540p eg (train/garden/bicycle, 9 runs -> 6 + 3)
run_dyn 0 tanks_and_temples/train 0 960 540 eg r56_train_540p_eg_s0 &
run_dyn 1 tanks_and_temples/train 1 960 540 eg r56_train_540p_eg_s1 &
run_dyn 2 tanks_and_temples/train 2 960 540 eg r56_train_540p_eg_s2 &
run_dyn 3 mipnerf360/garden 0 960 540 eg        r56_garden_540p_eg_s0 &
run_dyn 4 mipnerf360/garden 1 960 540 eg        r56_garden_540p_eg_s1 &
run_dyn 5 mipnerf360/garden 2 960 540 eg        r56_garden_540p_eg_s2 &
run_dyn 6 mipnerf360/bicycle 0 960 540 eg       r56_bicycle_540p_eg_s0 &
wait
echo WAVE1A_DONE
run_dyn 0 mipnerf360/bicycle 1 960 540 eg       r56_bicycle_540p_eg_s1 &
run_dyn 1 mipnerf360/bicycle 2 960 540 eg       r56_bicycle_540p_eg_s2 &
# wave 1b: 540p full (9 runs)
run_dyn 2 tanks_and_temples/train 0 960 540 full r56_train_540p_full_s0 &
run_dyn 3 tanks_and_temples/train 1 960 540 full r56_train_540p_full_s1 &
run_dyn 4 tanks_and_temples/train 2 960 540 full r56_train_540p_full_s2 &
run_dyn 5 mipnerf360/garden 0 960 540 full       r56_garden_540p_full_s0 &
wait
echo WAVE1B_DONE
run_dyn 0 mipnerf360/garden 1 960 540 full       r56_garden_540p_full_s1 &
run_dyn 1 mipnerf360/garden 2 960 540 full       r56_garden_540p_full_s2 &
run_dyn 2 mipnerf360/bicycle 0 960 540 full      r56_bicycle_540p_full_s0 &
run_dyn 3 mipnerf360/bicycle 1 960 540 full      r56_bicycle_540p_full_s1 &
run_dyn 4 mipnerf360/bicycle 2 960 540 full      r56_bicycle_540p_full_s2 &
wait
echo WAVE1C_DONE
# wave 2: 720p eg (9 runs)
run_dyn 0 tanks_and_temples/train 0 1280 720 eg r56_train_720p_eg_s0 &
run_dyn 1 tanks_and_temples/train 1 1280 720 eg r56_train_720p_eg_s1 &
run_dyn 2 tanks_and_temples/train 2 1280 720 eg r56_train_720p_eg_s2 &
run_dyn 3 mipnerf360/garden 0 1280 720 eg        r56_garden_720p_eg_s0 &
run_dyn 4 mipnerf360/garden 1 1280 720 eg        r56_garden_720p_eg_s1 &
run_dyn 5 mipnerf360/garden 2 1280 720 eg        r56_garden_720p_eg_s2 &
run_dyn 6 mipnerf360/bicycle 0 1280 720 eg       r56_bicycle_720p_eg_s0 &
wait
echo WAVE2A_DONE
run_dyn 0 mipnerf360/bicycle 1 1280 720 eg       r56_bicycle_720p_eg_s1 &
run_dyn 1 mipnerf360/bicycle 2 1280 720 eg       r56_bicycle_720p_eg_s2 &
# wave 2b: 720p full (9 runs)
run_dyn 2 tanks_and_temples/train 0 1280 720 full r56_train_720p_full_s0 &
run_dyn 3 tanks_and_temples/train 1 1280 720 full r56_train_720p_full_s1 &
run_dyn 4 tanks_and_temples/train 2 1280 720 full r56_train_720p_full_s2 &
run_dyn 5 mipnerf360/garden 0 1280 720 full       r56_garden_720p_full_s0 &
wait
echo WAVE2B_DONE
run_dyn 0 mipnerf360/garden 1 1280 720 full       r56_garden_720p_full_s1 &
run_dyn 1 mipnerf360/garden 2 1280 720 full       r56_garden_720p_full_s2 &
run_dyn 2 mipnerf360/bicycle 0 1280 720 full      r56_bicycle_720p_full_s0 &
run_dyn 3 mipnerf360/bicycle 1 1280 720 full      r56_bicycle_720p_full_s1 &
run_dyn 4 mipnerf360/bicycle 2 1280 720 full      r56_bicycle_720p_full_s2 &
wait
echo ALL_DONE_R56