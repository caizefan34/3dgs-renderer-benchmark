#!/bin/bash
# Round 43: error_alpha sweep on the high-N scenes (garden / bicycle).
#
# M4/M5 found the sole remaining quality gap on high-N scenes (garden PSNR
# -0.76 dB / LPIPS +0.050 at 2.12x; bicycle LPIPS +0.050 at 1.98x). The
# error-guided sampler (p ~ error^alpha, with-replacement) concentrates draws
# on high-error tiles; lower alpha flattens the distribution toward uniform,
# raising unique-tile coverage (sr -> r) at the cost of a little selectivity.
# This sweep tests whether alpha in {0.5, 0.75} closes the high-N gap while
# staying at/above the 1.8x target. Full references already measured
# (round-41d bicycle, round-42 garden); only eg runs are launched.
#
# Protocol: same R36 recipe + recommended op point as round-41d/42
# (error_guided r=0.35 lambda=0.7 + --lpips-full-res, 3000 steps, 1920x1080,
# n-train 4 / n-eval 3), 3 seeds per alpha.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=${OUT:-/tmp/qsR43}
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 alpha=$3 seed=$4 tag=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha $alpha \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

run_garden_a05 () {
  run_dyn 0 mipnerf360/garden 0.5 0 r43_garden_a05_s0
  run_dyn 0 mipnerf360/garden 0.5 1 r43_garden_a05_s1
  run_dyn 0 mipnerf360/garden 0.5 2 r43_garden_a05_s2
}
run_garden_a075 () {
  run_dyn 1 mipnerf360/garden 0.75 0 r43_garden_a075_s0
  run_dyn 1 mipnerf360/garden 0.75 1 r43_garden_a075_s1
  run_dyn 1 mipnerf360/garden 0.75 2 r43_garden_a075_s2
}
run_bicycle_a05 () {
  run_dyn 2 mipnerf360/bicycle 0.5 0 r43_bicycle_a05_s0
  run_dyn 2 mipnerf360/bicycle 0.5 1 r43_bicycle_a05_s1
  run_dyn 2 mipnerf360/bicycle 0.5 2 r43_bicycle_a05_s2
}
run_garden_a05 &
run_garden_a075 &
run_bicycle_a05 &
wait
echo ALL_DONE_R43
