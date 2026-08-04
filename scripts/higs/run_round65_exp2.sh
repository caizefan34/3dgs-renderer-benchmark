#!/bin/bash
# Round-65 exp2: extend prog-res x decay cell to truck (high-N) + bonsai (mid-N).
#   truck:  full+MA ctrl s0/s1/s2 + prog+MA+decay s0/s1/s2 (GPUs 0-5)
#   bonsai: full+MA ctrl s0/s1/s2 + prog+MA+decay s0/s1/s2 (GPUs 0-5, wave 2)
# Same recipe as R65 exp1 (anchor-densify-every-2 for both scenes).
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR65exp2
mkdir -p $OUT

run () {
  local gpu=$1 scene=$2 seed=$3 arm=$4
  local extra="--anchor-densify --anchor-densify-every 2"
  local dflag=""; local rflag=""
  if [ "$arm" = "pd" ]; then
    dflag="--masked-adam-union-decay 0.99 --masked-adam-union-decay-eval-proj"
    rflag="--res-schedule 0.5:0,1.0:1500"
  fi
  local sname=$(basename "$scene")
  local tag="${sname}_${arm}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval 1 --masked-adam $dflag $rflag \
    $extra --out $OUT/$tag.json > $OUT/$tag.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}
run 0 tanks_and_temples/truck 0 ctrl &
run 1 tanks_and_temples/truck 1 ctrl &
run 2 tanks_and_temples/truck 2 ctrl &
run 3 tanks_and_temples/truck 0 pd &
run 4 tanks_and_temples/truck 1 pd &
run 5 tanks_and_temples/truck 2 pd &
wait
echo WAVE1_TRUCK_DONE
run 0 mipnerf360/bonsai 0 ctrl &
run 1 mipnerf360/bonsai 1 ctrl &
run 2 mipnerf360/bonsai 2 ctrl &
run 3 mipnerf360/bonsai 0 pd &
run 4 mipnerf360/bonsai 1 pd &
run 5 mipnerf360/bonsai 2 pd &
wait
echo WAVE2_BONSAI_DONE
echo R65EXP2_DONE
cat $OUT/rc.txt