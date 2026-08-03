#!/bin/bash
# Round 41b: M4 final gate - A100 multi-seed retest at the 1.8x operating point.
#
# Background (EPIC-05 A100-80GB, 2026-08-04, torch 2.7.0+cu128, gsplat 1.5.3
# built with GSPLAT_SKIP_FROM_WORLD=1): paired same-session timing
#   full r=1.0 27.4ms -> eg r=0.35 (sr~0.27) 16.7ms (1.64x)
#   -> eg r=0.30 (sr~0.24) 14.9ms (1.84x) -> eg r=0.25 (sr~0.21) 14.2ms (1.93x)
# so the 1.8x point on A100 sits at nominal r ~= 0.30-0.35.
#
# Protocol: R36 recipe (lr-decay 0.1 + densify-window 1500 + LPIPS w=0.1 every 25
# + error_guided refresh 25), 3000 steps, 1920x1080, n-train 4 / n-eval 3.
# train 3 seeds full vs eg r=0.35 vs eg r=0.30; bicycle seed 0 same 3 configs.
# Sequential on GPU 0 so per-step timing stays contention-free.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=${OUT:-/tmp/qsR41}
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 ratio=$3 seed=$4 tag=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio $ratio --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 1.0 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

# sequential on GPU 0: timing contention-free
run_dyn 0 tanks_and_temples/train 1.0  0 m4_train_full_s0
run_dyn 0 tanks_and_temples/train 1.0  1 m4_train_full_s1
run_dyn 0 tanks_and_temples/train 1.0  2 m4_train_full_s2
run_dyn 0 tanks_and_temples/train 0.35 0 m4_train_eg035_s0
run_dyn 0 tanks_and_temples/train 0.35 1 m4_train_eg035_s1
run_dyn 0 tanks_and_temples/train 0.35 2 m4_train_eg035_s2
run_dyn 0 tanks_and_temples/train 0.30 0 m4_train_eg030_s0
run_dyn 0 tanks_and_temples/train 0.30 1 m4_train_eg030_s1
run_dyn 0 tanks_and_temples/train 0.30 2 m4_train_eg030_s2
run_dyn 0 mipnerf360/bicycle 1.0  0 m4_bicycle_full_s0
run_dyn 0 mipnerf360/bicycle 0.35 0 m4_bicycle_eg035_s0
run_dyn 0 mipnerf360/bicycle 0.30 0 m4_bicycle_eg030_s0
echo ALL_DONE
