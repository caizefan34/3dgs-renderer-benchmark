#!/bin/bash
# Round-65 exp1: progressive-res (0.5:0,1.0:1500) x masked-Adam +/- decay+proj-refresh.
#   bicycle: prog+MA ctrl s0/s1/s2 + prog+MA+decay s0/s1/s2 (GPUs 0-5)
#   garden:  prog+MA ctrl s0/s1/s2 + prog+MA+decay s0/s1/s2 (GPUs 0-5, wave 2)
# Recipe = R64 exp2 + --res-schedule; decay arm = d0.99 + projection refresh.
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR65exp1
mkdir -p $OUT

run () {
  local gpu=$1 scene=$2 seed=$3 decay=$4
  local extra="--anchor-densify --anchor-densify-every 2"
  local dflag=""
  if [ "$decay" = "1" ]; then dflag="--masked-adam-union-decay 0.99 --masked-adam-union-decay-eval-proj"; fi
  local sname=$(basename "$scene")
  local tag="${sname}_progma_${decay}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --res-schedule "0.5:0,1.0:1500" \
    --cull-interval 1 --masked-adam $dflag \
    $extra --out $OUT/$tag.json > $OUT/$tag.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}
run 0 mipnerf360/bicycle 0 0 &
run 1 mipnerf360/bicycle 1 0 &
run 2 mipnerf360/bicycle 2 0 &
run 3 mipnerf360/bicycle 0 1 &
run 4 mipnerf360/bicycle 1 1 &
run 5 mipnerf360/bicycle 2 1 &
wait
echo WAVE1_BICYCLE_DONE
run 0 mipnerf360/garden 0 0 &
run 1 mipnerf360/garden 1 0 &
run 2 mipnerf360/garden 2 0 &
run 3 mipnerf360/garden 0 1 &
run 4 mipnerf360/garden 1 1 &
run 5 mipnerf360/garden 2 1 &
wait
echo WAVE2_GARDEN_DONE
echo R65EXP1_DONE
cat $OUT/rc.txt