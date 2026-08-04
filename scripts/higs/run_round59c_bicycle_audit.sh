#!/bin/bash
# Round 59 bicycle variance audit: 3 extra seeds (3,4,5) x {k1, k4, g16} on
# the critical high-parallax scene to separate mask-staleness effects from the
# scene's known run-to-run variance (the gated arm's seed-0 collapsed during
# its phase-1, which uses K=1 exactly like the baseline, flagging inherent
# variance). 9 runs, same 720p eg recipe.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR59c
mkdir -p $OUT

run_dyn () {
  local gpu=$1 seed=$2 K=$3 tag=$4
  local extra="--anchor-densify --anchor-densify-every 2"
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene mipnerf360/bicycle \
    --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 3000 \
    --width 1280 --height 720 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval $K \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

run_g16 () {
  local gpu=$1 seed=$2 tag=$3
  local extra="--anchor-densify --anchor-densify-every 2"
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene mipnerf360/bicycle \
    --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 3000 \
    --width 1280 --height 720 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval-schedule "1:0,16:1500" \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

jobs=0
for seed in 3 4 5; do
  gpu=$((jobs % 6)); run_dyn $gpu $seed 1 "r59_bicycle_720p_k1_s${seed}" & jobs=$((jobs+1))
  gpu=$((jobs % 6)); run_dyn $gpu $seed 4 "r59_bicycle_720p_k4_s${seed}" & jobs=$((jobs+1))
  gpu=$((jobs % 6)); run_g16 $gpu $seed "r59_bicycle_720p_g16_s${seed}" & jobs=$((jobs+1))
  if [ $((jobs % 6)) -eq 0 ]; then wait; echo "WAVE $((jobs / 6))_DONE"; fi
done
wait
echo ALL_DONE_R59C