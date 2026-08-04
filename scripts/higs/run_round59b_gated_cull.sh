#!/bin/bash
# Round 59 follow-up: phase-gated cull cache validation.
# The uniform k4/k16 sweep degraded bicycle PSNR exactly after the densify
# window (step 1500) ended, i.e. in the regime where the stale mask is no
# longer refreshed by densify. This arm gates the cache to the post-window
# phase: cull every step while densify is active (mask cadence already forced
# by mark_dirty), K=16 after. 3 scenes x 3 seeds = 9 runs, same 720p eg
# recipe as the round-59 sweep.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR59b
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 tag=$4
  local extra=""
  if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1280 --height 720 --seed $seed \
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
for scene in tanks_and_temples/train mipnerf360/garden mipnerf360/bicycle; do
  for seed in 0 1 2; do
    sname=$(basename "$scene")
    tag="r59_${sname}_720p_g16_s${seed}"
    gpu=$((jobs % 6))
    run_dyn $gpu "$scene" $seed $tag &
    jobs=$((jobs + 1))
    if [ $((jobs % 6)) -eq 0 ]; then wait; echo "WAVE $((jobs / 6))_DONE"; fi
  done
done
wait
echo ALL_DONE_R59B