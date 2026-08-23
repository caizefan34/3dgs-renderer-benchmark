#!/bin/bash
# Round 55: progressive-resolution + full-res LPIPS signal only.
# R52b (--res-schedule-full-signal) forced BOTH the LPIPS steps and the
# anchor-densify steps full-res during the coarse stage; the full/stage-res
# densify alternation (every 2 events) destabilized high-N scenes. R55
# isolates the perceptual lever: --res-schedule-full-lpips keeps only the
# LPIPS steps at the full target resolution during the coarse stage while
# anchor densify stays at stage scale. Recipe otherwise identical to R52:
# error_guided r=0.35 lambda=0.7 + full-res LPIPS every 25; high-N scenes +
# anchor densify every 2, train no anchor. 3 seeds per scene.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR55
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 anchor=$4 tag=$5
  local extra=""
  if [ "$anchor" = "1" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --res-schedule "0.5:0,1.0:1500" --res-schedule-full-lpips \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# wave 1: train (no anchor) + garden (anchor)
run_dyn 0 tanks_and_temples/train 0 0 r55_train_fulllpips_s0 &
run_dyn 1 tanks_and_temples/train 1 0 r55_train_fulllpips_s1 &
run_dyn 2 tanks_and_temples/train 2 0 r55_train_fulllpips_s2 &
run_dyn 3 mipnerf360/garden 0 1 r55_garden_fulllpips_s0 &
run_dyn 4 mipnerf360/garden 1 1 r55_garden_fulllpips_s1 &
run_dyn 5 mipnerf360/garden 2 1 r55_garden_fulllpips_s2 &
wait
echo WAVE1_DONE
# wave 2: bicycle (anchor)
run_dyn 0 mipnerf360/bicycle 0 1 r55_bicycle_fulllpips_s0 &
run_dyn 1 mipnerf360/bicycle 1 1 r55_bicycle_fulllpips_s1 &
run_dyn 2 mipnerf360/bicycle 2 1 r55_bicycle_fulllpips_s2 &
wait
echo ALL_DONE_R55