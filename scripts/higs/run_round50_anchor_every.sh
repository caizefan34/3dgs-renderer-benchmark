#!/bin/bash
# Round 50 (3-seed confirmation): --anchor-densify-every 2 - subsample full-res
# densify steps on the high-N scenes.
#
# Round-50 single-seed screening (vs round-42/41d eg baselines):
#   garden: every2 18.127 / 0.4445 / 20.89ms vs anchor1 18.178 / 0.4345 / 21.65
#   every4 collapses back to eg (17.951 / 0.4477) -> every2 keeps most of the
#   anchor PSNR gain at ~half the cost; bicycle within scene noise either way.
# Seeds 1/2 here (seed 0 already in results/higs-round50/); same recommended op
# point otherwise (r=0.35 lambda=0.7 + full-res LPIPS + anchor-densify, 3000
# steps, 1920x1080, n-train 4 / n-eval 3).
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR50c
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --anchor-densify --anchor-densify-every 2 \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

run_dyn 2 mipnerf360/garden 1 r50_garden_anchor_every2_s1
run_dyn 3 mipnerf360/garden 2 r50_garden_anchor_every2_s2
run_dyn 4 mipnerf360/bicycle 1 r50_bicycle_anchor_every2_s1
run_dyn 5 mipnerf360/bicycle 2 r50_bicycle_anchor_every2_s2
echo ALL_DONE_R50C
