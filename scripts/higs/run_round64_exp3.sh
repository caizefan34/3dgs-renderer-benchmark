#!/bin/bash
# Round-64 exp3: 1080p quality-max cell, decay-rate 0.999 vs 0.99 (garden/bicycle).
#   garden:  0.99 s0 (wave anchor) + 0.999 s0/s1/s2  (GPUs 0-3)
#   bicycle: 0.99 s0 (wave anchor) + 0.999 s0/s1/s2  (GPUs 4-7)
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR64exp3
mkdir -p $OUT

run () {
  local gpu=$1 scene=$2 seed=$3 decay=$4
  local extra="--anchor-densify --anchor-densify-every 2"
  local sname=$(basename "$scene")
  local tag="${sname}_${decay}_1_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval 1 --masked-adam --masked-adam-union-decay $decay --masked-adam-union-decay-eval-proj \
    $extra --out $OUT/$tag.json > $OUT/$tag.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}
run 0 mipnerf360/garden 0 0.99 &
run 1 mipnerf360/garden 0 0.999 &
run 2 mipnerf360/garden 1 0.999 &
run 3 mipnerf360/garden 2 0.999 &
run 4 mipnerf360/bicycle 0 0.99 &
run 5 mipnerf360/bicycle 0 0.999 &
run 6 mipnerf360/bicycle 1 0.999 &
run 7 mipnerf360/bicycle 2 0.999 &
wait
echo EXP3_DONE
cat $OUT/rc.txt