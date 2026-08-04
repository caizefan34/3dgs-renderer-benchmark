#!/bin/bash
# Round 42: M5 multi-scene matrix - extend the M4 1.8x operating-point evidence
# to the remaining benchmark scenes (garden / bonsai / truck) on EPIC-05 A100.
#
# M4 (rounds 41b-41d) certified the recommended operating point on train
# (1.82x, PSNR +0.40 dB) and bicycle (1.98x, PSNR parity; LPIPS +0.050 the sole
# honest bound). M5 checks the same op point holds quality parity on the other
# scenes: mipnerf360/garden (5.8M), mipnerf360/bonsai (1.2M),
# tanks_and_temples/truck (2.5M).
#
# Protocol: identical to round-41d M4 (R36 recipe: lr-decay 0.1 +
# densify-window 1500 + LPIPS w=0.1 every 25 + error_guided alpha=1.0 refresh
# 25 + eval every 300; 3000 steps; 1920x1080; n-train 4 / n-eval 3), full
# r=1.0 vs eg r=0.35 lambda=0.7 + --lpips-full-res, 3 seeds each.
# Parallel: one scene per GPU (garden/bonsai/truck on GPUs 0/1/2), 6 sequential runs each (runs at 07:19-08:14
# local showed ~2-6 min per 3000-step run on A100).
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=${OUT:-/tmp/qsR42}
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 ratio=$3 lam=$4 fr=$5 seed=$6 tag=$7
  local FRARG=""
  if [ "$fr" = "1" ]; then FRARG="--lpips-full-res"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio $ratio --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda $lam --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 $FRARG \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# garden (gpu 0): full 3-seed + eg r=0.35 l=0.7 fr 3-seed
run_garden () {
  run_dyn 0 mipnerf360/garden 1.0  1.0 0 0 m5_garden_full_s0
  run_dyn 0 mipnerf360/garden 1.0  1.0 0 1 m5_garden_full_s1
  run_dyn 0 mipnerf360/garden 1.0  1.0 0 2 m5_garden_full_s2
  run_dyn 0 mipnerf360/garden 0.35 0.7 1 0 m5_garden_eg035_l07_fr_s0
  run_dyn 0 mipnerf360/garden 0.35 0.7 1 1 m5_garden_eg035_l07_fr_s1
  run_dyn 0 mipnerf360/garden 0.35 0.7 1 2 m5_garden_eg035_l07_fr_s2
}
# bonsai (gpu 1)
run_bonsai () {
  run_dyn 1 mipnerf360/bonsai 1.0  1.0 0 0 m5_bonsai_full_s0
  run_dyn 1 mipnerf360/bonsai 1.0  1.0 0 1 m5_bonsai_full_s1
  run_dyn 1 mipnerf360/bonsai 1.0  1.0 0 2 m5_bonsai_full_s2
  run_dyn 1 mipnerf360/bonsai 0.35 0.7 1 0 m5_bonsai_eg035_l07_fr_s0
  run_dyn 1 mipnerf360/bonsai 0.35 0.7 1 1 m5_bonsai_eg035_l07_fr_s1
  run_dyn 1 mipnerf360/bonsai 0.35 0.7 1 2 m5_bonsai_eg035_l07_fr_s2
}
# truck (gpu 2)
run_truck () {
  run_dyn 2 tanks_and_temples/truck 1.0  1.0 0 0 m5_truck_full_s0
  run_dyn 2 tanks_and_temples/truck 1.0  1.0 0 1 m5_truck_full_s1
  run_dyn 2 tanks_and_temples/truck 1.0  1.0 0 2 m5_truck_full_s2
  run_dyn 2 tanks_and_temples/truck 0.35 0.7 1 0 m5_truck_eg035_l07_fr_s0
  run_dyn 2 tanks_and_temples/truck 0.35 0.7 1 1 m5_truck_eg035_l07_fr_s1
  run_dyn 2 tanks_and_temples/truck 0.35 0.7 1 2 m5_truck_eg035_l07_fr_s2
}
# one scene per GPU, in parallel; each scene block is 6 sequential runs
run_garden &
run_bonsai &
run_truck &
wait
echo ALL_DONE_M5
