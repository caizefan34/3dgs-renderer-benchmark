#!/bin/bash
# Round-62 exp2: full 3000-step screen - 3 scenes x {decay 0.999, 0.99} x seed 0.
# Compare vs r60 (k1 masked-Adam) / r61 (ws256) seed-0 baselines.
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR62exp2
mkdir -p $OUT

run () {
  local gpu=$1 scene=$2 rate=$3
  local extra=""
  if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  local sname=$(basename "$scene")
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1280 --height 720 --seed 0 \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval 1 --masked-adam --masked-adam-union-decay $rate \
    $extra --out $OUT/${sname}_d${rate}_s0.json > $OUT/${sname}_d${rate}_s0.log 2>&1
  echo "DONE $sname d$rate rc=$?" >> $OUT/rc.txt
}
run 0 mipnerf360/garden 0.999 &
run 1 mipnerf360/garden 0.99 &
run 2 tanks_and_temples/train 0.999 &
run 3 tanks_and_temples/train 0.99 &
run 4 mipnerf360/bicycle 0.999 &
run 5 mipnerf360/bicycle 0.99 &
wait
echo EXP2_DONE
cat $OUT/rc.txt