#!/bin/bash
# Round 60 (trainable-HiGS speed lever): cull-masked Adam step.
# The masked-Adam kernel (benchmark/higs_masked_adam.py) runs the exact torch
# fused-Adam math only on rows whose train-forward union-visibility mask is
# True, so the optimizer memory traffic scales with the visible set instead
# of N (isolated probe at 3.5M gaussians: 4.18 ms fused -> 2.40 ms at 42%
# visible, 2.84 ms at 58%).  Sweep: 3 scenes x 3 seeds = 9 runs at k1
# (per-step fresh mask), 720p eg recipe.  Baseline = round-59 k1 runs
# (identical config; masked_adam=False path is bit-identical).  Measures the
# end-to-end wall impact (train_ms) including topology feedback (frozen
# out-of-view rows change densify/prune counts) and the quality delta
# (PSNR/LPIPS/SSIM).
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR60
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 tag=$4
  local extra=""
  if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts     --n-train 4 --n-eval 3 --steps 3000 --width 1280 --height 720 --seed $seed     --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0     --error-refresh-every 25 --error-lambda 0.7 --eval-every 300     --lr-decay 0.1 --densify-window 1500     --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res     --cull-interval 1 --masked-adam     $extra     --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}

jobs=0
for scene in tanks_and_temples/train mipnerf360/garden mipnerf360/bicycle; do
  for seed in 0 1 2; do
    sname=$(basename "$scene")
    tag="r60_${sname}_720p_ma_s${seed}"
    gpu=$((jobs % 6))
    run_dyn $gpu "$scene" $seed $tag &
    jobs=$((jobs + 1))
    if [ $((jobs % 6)) -eq 0 ]; then wait; echo "WAVE $((jobs / 6))_DONE"; fi
  done
done
wait
echo ALL_DONE_R60
