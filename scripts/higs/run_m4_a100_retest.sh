#!/bin/bash
# Round 41d: M4 final gate - A100 multi-seed retest at the 1.8x operating point
# with the recommended round-41d recipe (--lpips-full-res).
#
# Background (EPIC-05 A100-80GB, 2026-08-04, torch 2.7.0+cu128, gsplat 1.5.3
# built with GSPLAT_SKIP_FROM_WORLD=1): paired same-session timing
#   full r=1.0 27.4ms -> eg r=0.35 (sr~0.27) 16.7ms (1.64x)
#   -> eg r=0.30 (sr~0.24) 14.9ms (1.84x) -> eg r=0.25 (sr~0.21) 14.2ms (1.93x)
# so the 1.8x point on A100 sits at nominal r ~= 0.30-0.35.
#
# Round 41d results (3000 steps, 1920x1080, n-train 4 / n-eval 3):
#   train 3-seed: full 16.673/0.6267/0.3678/21.98ms vs eg r=0.35 l=0.7
#     17.074/0.6295/0.3870/12.06ms -> 1.82x (PSNR +0.40, LPIPS +0.019)
#   bicycle 3-run full: 16.024+-0.072/0.3908/0.4795/45.20ms vs eg r=0.35 l=0.7
#     + --lpips-full-res 3-seed: 15.965+-0.109/0.3891/0.5298/22.81ms -> 1.98x
#     (PSNR parity -0.06, LPIPS +0.050+-0.002 = the sole remaining honest bound)
#   lambda scan (same-session): lambda=0.7 is a sharp optimum on bicycle PSNR
#     (0.5/0.85 collapse ~1.2 dB); LPIPS is lambda-robust (~0.531).
#   Determinism: bicycle same-seed reruns vary ~+-0.1-0.3 dB (CUDA atomics
#     amplified by densify/prune) -> bicycle claims always multi-seed.
#   6000-step probe: R36 recipe degrades both sides after 3000 steps; the LPIPS
#     gap is asymptotic, not a convergence-rate artifact.
#
# Protocol: R36 recipe (lr-decay 0.1 + densify-window 1500 + LPIPS w=0.1 every
# 25 + error_guided refresh 25) + --lpips-full-res at the recommended op point.
# Sequential on GPU 0 so per-step timing stays contention-free.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=${OUT:-/tmp/qsR41d}
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

# sequential on GPU 0: timing contention-free
# train: full 3-seed + recommended op point 3-seed
run_dyn 0 tanks_and_temples/train 1.0  1.0 0 0 m4_train_full_s0
run_dyn 0 tanks_and_temples/train 1.0  1.0 0 1 m4_train_full_s1
run_dyn 0 tanks_and_temples/train 1.0  1.0 0 2 m4_train_full_s2
run_dyn 0 tanks_and_temples/train 0.35 0.7 1 0 m4_train_eg035_l07_fr_s0
run_dyn 0 tanks_and_temples/train 0.35 0.7 1 1 m4_train_eg035_l07_fr_s1
run_dyn 0 tanks_and_temples/train 0.35 0.7 1 2 m4_train_eg035_l07_fr_s2
# bicycle: full 3-run + recommended op point 3-seed
run_dyn 0 mipnerf360/bicycle 1.0  1.0 0 0 m4_bicycle_full_s0
run_dyn 0 mipnerf360/bicycle 1.0  1.0 0 1 m4_bicycle_full_s1
run_dyn 0 mipnerf360/bicycle 1.0  1.0 0 2 m4_bicycle_full_s2
run_dyn 0 mipnerf360/bicycle 0.35 0.7 1 0 m4_bicycle_eg035_l07_fr_s0
run_dyn 0 mipnerf360/bicycle 0.35 0.7 1 1 m4_bicycle_eg035_l07_fr_s1
run_dyn 0 mipnerf360/bicycle 0.35 0.7 1 2 m4_bicycle_eg035_l07_fr_s2
echo ALL_DONE
