#!/bin/bash
# Round-64 exp2: 1080p quality-max cell, cross-scene (bicycle/train).
#   bicycle: ctrl s0 + stack s0/s1/s2 (GPUs 0-3)
#   train:   ctrl s0 + stack s0/s1/s2 (GPUs 4-7)
# Config matches R62 exp2 scene conventions (anchor-densify on non-train scenes).
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR64exp2
mkdir -p $OUT

run () {
  local gpu=$1 scene=$2 seed=$3 decay=$4 proj=$5
  local extra=""
  if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  local sname=$(basename "$scene")
  local pflag=""
  if [ "$proj" = "1" ]; then pflag="--masked-adam-union-decay-eval-proj"; fi
  local tag="${sname}_${decay}_${proj}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval 1 --masked-adam --masked-adam-union-decay $decay $pflag \
    $extra --out $OUT/$tag.json > $OUT/$tag.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}
run 0 mipnerf360/bicycle 0 0.0 0 &
run 1 mipnerf360/bicycle 0 0.99 1 &
run 2 mipnerf360/bicycle 1 0.99 1 &
run 3 mipnerf360/bicycle 2 0.99 1 &
run 4 tanks_and_temples/train 0 0.0 0 &
run 5 tanks_and_temples/train 0 0.99 1 &
run 6 tanks_and_temples/train 1 0.99 1 &
run 7 tanks_and_temples/train 2 0.99 1 &
wait
echo EXP2_DONE
cat $OUT/rc.txt
