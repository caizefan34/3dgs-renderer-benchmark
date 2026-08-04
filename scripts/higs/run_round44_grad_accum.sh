#!/bin/bash
# Round 44: densify grad-accum lever on the high-N scenes (garden / bicycle).
#
# Round-43 closed alpha as a lever: the high-N quality gap (garden LPIPS
# +0.048-0.050, bicycle +0.045-0.050 at 1.74-2.12x) is not a sampling-
# concentration artifact. Round 44 tests the remaining structural difference
# vs the full-resolution pipeline: densify decisions. Under tile-sampled
# training (r=0.35) the instantaneous per-step position-gradient norm
# under-counts Gaussians outside sampled tiles, so dup/clone fires on a
# sparse signal. --densify-grad-accum accumulates (detached) position-
# gradient norms over the densify window (standard 3DGS recipe) and drives
# the densify decision from the accumulated signal.
#
# Protocol: identical to round-41d/42 (R36 recipe + recommended op point
# error_guided r=0.35 lambda=0.7 + --lpips-full-res, 3000 steps, 1920x1080,
# n-train 4 / n-eval 3), 3 seeds per scene, eg runs only (full references
# already measured in round-41d/42). Pure-Python change: no CUDA rebuild.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=${OUT:-/tmp/qsR44}
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --densify-grad-accum \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

run_garden () {
  run_dyn 0 mipnerf360/garden 0 r44_garden_eg035_l07_fr_ga_s0
  run_dyn 0 mipnerf360/garden 1 r44_garden_eg035_l07_fr_ga_s1
  run_dyn 0 mipnerf360/garden 2 r44_garden_eg035_l07_fr_ga_s2
}
run_bicycle () {
  run_dyn 1 mipnerf360/bicycle 0 r44_bicycle_eg035_l07_fr_ga_s0
  run_dyn 1 mipnerf360/bicycle 1 r44_bicycle_eg035_l07_fr_ga_s1
  run_dyn 1 mipnerf360/bicycle 2 r44_bicycle_eg035_l07_fr_ga_s2
}
run_garden &
run_bicycle &
wait
echo ALL_DONE_R44