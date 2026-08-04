#!/bin/bash
# Round 59 bicycle variance audit (part 2): k16 seeds 3-5 for a symmetric
# n=6 table on the critical scene.
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR59d
mkdir -p $OUT
run_dyn () {
  local gpu=$1 seed=$2 tag=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene mipnerf360/bicycle \
    --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 3000 \
    --width 1280 --height 720 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval 16 --anchor-densify --anchor-densify-every 2 \
    --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?"
}
run_dyn 0 3 r59_bicycle_720p_k16_s3 &
run_dyn 1 4 r59_bicycle_720p_k16_s4 &
run_dyn 2 5 r59_bicycle_720p_k16_s5 &
wait
echo ALL_DONE_R59D