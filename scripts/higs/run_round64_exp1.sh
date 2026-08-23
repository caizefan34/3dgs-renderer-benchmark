#!/bin/bash
# Round-64 exp1: 1080p quality-max cell - does the R63 decay+proj lever stack with the
# recommended quality-max config (1080p eg + anchor-densify-every-2 + masked-adam)?
#   GPU0-2: ctrl1080 (R60 op point at 1080p, seed 0/1/2)
#   GPU3-5: stack1080 = ctrl1080 + decay 0.99 + proj refresh (seed 0/1/2)
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR64exp1
mkdir -p $OUT

run () {
  local gpu=$1 seed=$2 decay=$3 proj=$4
  local pflag=""
  if [ "$proj" = "1" ]; then pflag="--masked-adam-union-decay-eval-proj"; fi
  local tag="garden_${decay}_${proj}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --anchor-densify --anchor-densify-every 2 \
    --cull-interval 1 --masked-adam --masked-adam-union-decay $decay $pflag \
    --out $OUT/$tag.json > $OUT/$tag.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}
run 0 0 0.0 0 &
run 1 1 0.0 0 &
run 2 2 0.0 0 &
run 3 0 0.99 1 &
run 4 1 0.99 1 &
run 5 2 0.99 1 &
wait
echo EXP1_DONE
cat $OUT/rc.txt
