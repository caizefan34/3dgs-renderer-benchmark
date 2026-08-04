#!/bin/bash
# Round 45: full-resolution densify signal probe on the high-N scenes.
#
# Round-43 (alpha) and round-44 (densify grad-accum) closed the sampling-side
# and accumulation-side levers: at r=0.35 the per-step position gradient is
# dominated by which tiles were sampled, so neither alpha nor 5-step norm
# accumulation reproduces the full-resolution densify signal (round-44 even
# over-densified bicycle and hurt quality). Round 45 gives densify decisions
# the actual full-resolution gradient at ZERO extra cost: densify_every=25
# aligns every densify step with the full-res LPIPS step (--lpips-loss-every
# 25 --lpips-full-res replaces that sampled-tile step with a full render), so
# the dup/clone decision runs on the full-frame gradient. Tests whether the
# high-N quality gap (garden/bicycle) originates in the densify side.
#
# Protocol: identical to round-41d/42 except densify-every 25 (was 5):
# R36 recipe + recommended op point error_guided r=0.35 lambda=0.7 +
# --lpips-full-res, 3000 steps, 1920x1080, n-train 4 / n-eval 3, 3 seeds.
# Wall-clock is unchanged (full-res steps are identical to the baseline).
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=${OUT:-/tmp/qsR45}
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 --densify-every 25 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

run_garden () {
  run_dyn 0 mipnerf360/garden 0 r45_garden_eg035_l07_fr_de25_s0
  run_dyn 0 mipnerf360/garden 1 r45_garden_eg035_l07_fr_de25_s1
  run_dyn 0 mipnerf360/garden 2 r45_garden_eg035_l07_fr_de25_s2
}
run_bicycle () {
  run_dyn 1 mipnerf360/bicycle 0 r45_bicycle_eg035_l07_fr_de25_s0
  run_dyn 1 mipnerf360/bicycle 1 r45_bicycle_eg035_l07_fr_de25_s1
  run_dyn 1 mipnerf360/bicycle 2 r45_bicycle_eg035_l07_fr_de25_s2
}
run_garden &
run_bicycle &
wait
echo ALL_DONE_R45