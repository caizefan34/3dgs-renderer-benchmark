#!/bin/bash
# Round 51 (M6 baseline 1/3): ICCV random-tile loss baseline - uniform random
# tile sampling with masked tile loss, vs error_guided at the SAME recommended
# op point (full 3000-step protocol, 1920x1080, n-train 4 / n-eval 3).
#
# Design: sampling strategy is the only variable. High-N scenes (garden,
# bicycle) use the round-50 recommended op point (full-res LPIPS + anchor
# densify every 2); train (low-N) uses the M4 op point (no anchor, per the
# round-47 opt-in guidance). train eg reruns restore the per-run files the
# round-41b summary lost (gitignored), so r51-summary.json is self-contained.
# 3 seeds per scene for both arms.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR51
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 mode=$4 anchor=$5 tag=$6
  local extra=""
  if [ "$anchor" = "1" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode $mode --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# wave 1: train both arms (no anchor)
run_dyn 0 tanks_and_temples/train 0 uniform 0 r51_train_uniform_s0 &
run_dyn 1 tanks_and_temples/train 1 uniform 0 r51_train_uniform_s1 &
run_dyn 2 tanks_and_temples/train 2 uniform 0 r51_train_uniform_s2 &
run_dyn 3 tanks_and_temples/train 0 error_guided 0 r51_train_eg_s0 &
run_dyn 4 tanks_and_temples/train 1 error_guided 0 r51_train_eg_s1 &
run_dyn 5 tanks_and_temples/train 2 error_guided 0 r51_train_eg_s2 &
wait
echo WAVE1_DONE
# wave 2: high-N both arms (anchor every 2)
run_dyn 0 mipnerf360/garden 0 uniform 1 r51_garden_uniform_s0 &
run_dyn 1 mipnerf360/garden 1 uniform 1 r51_garden_uniform_s1 &
run_dyn 2 mipnerf360/garden 2 uniform 1 r51_garden_uniform_s2 &
run_dyn 3 mipnerf360/bicycle 0 uniform 1 r51_bicycle_uniform_s0 &
run_dyn 4 mipnerf360/bicycle 1 uniform 1 r51_bicycle_uniform_s1 &
run_dyn 5 mipnerf360/bicycle 2 uniform 1 r51_bicycle_uniform_s2 &
wait
echo ALL_DONE_R51
