#!/bin/bash
# Round 47 (screening): anchor-densify diagnostic - full-res densify steps at
# the original cadence (densify-every 5) on the high-N scenes.
#
# Rounds 43/44 closed sampling-side and accumulation-side levers; round-45
# showed the coarse densify cadence (every 25) degrades quality even with a
# full-res signal, so the cadence itself matters. --anchor-densify keeps the
# original 5-step cadence but runs each densify step at full resolution,
# giving dup/clone the true full-frame position gradient (cost ~1.2x wall).
# Screening: 1 seed per scene first; if the direction is promising, extend
# to 3 seeds. Same recommended op point otherwise (r=0.35 lambda=0.7 +
# full-res LPIPS, R36 recipe, 3000 steps, 1920x1080, n-train 4 / n-eval 3).
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=${OUT:-/tmp/qsR47}
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --anchor-densify \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

run_dyn 2 mipnerf360/garden 0 r47_garden_eg035_l07_fr_anchor_s0
run_dyn 3 mipnerf360/bicycle 0 r47_bicycle_eg035_l07_fr_anchor_s0
echo ALL_DONE_R47