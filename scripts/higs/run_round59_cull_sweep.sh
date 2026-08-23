#!/bin/bash
# Round 59 (speed-max enhancement): cull-mask refresh interval sweep.
# Profiling (round-59 profiler) showed the per-step batched full-N culling
# projection is a pixel-independent floor cost; the benchmark fixes cam_ids
# per phase so visibility drifts only via parameter updates -> a camera-set
# keyed cull cache (--cull-interval K, patched gsplat) can skip recomputing
# the mask for K-1 steps. Sweep K in {1,4,16} x {train,garden,bicycle} x 3
# seeds = 27 runs on the standard 720p eg recipe (R56/R58); R56 720p eg/full
# are reused as historical references (K=1 == pre-patch behavior).
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR59
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 K=$4 tag=$5
  local extra=""
  if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1280 --height 720 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval $K \
    $extra \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}

jobs=0
for scene in tanks_and_temples/train mipnerf360/garden mipnerf360/bicycle; do
  for K in 1 4 16; do
    for seed in 0 1 2; do
      sname=$(basename "$scene")
      tag="r59_${sname}_720p_k${K}_s${seed}"
      gpu=$((jobs % 6))
      run_dyn $gpu "$scene" $seed $K $tag &
      jobs=$((jobs + 1))
      if [ $((jobs % 6)) -eq 0 ]; then wait; echo "WAVE $((jobs / 6))_DONE"; fi
    done
  done
done
wait
echo ALL_DONE_R59