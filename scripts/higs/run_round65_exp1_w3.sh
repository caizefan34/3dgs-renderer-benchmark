#!/bin/bash
# Round-65 exp1 wave 3: train (no anchor) progressive-res x masked-Adam +/- decay.
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR65exp1
run () {
  local gpu=$1 seed=$2 decay=$3
  local dflag=""
  if [ "$decay" = "1" ]; then dflag="--masked-adam-union-decay 0.99 --masked-adam-union-decay-eval-proj"; fi
  local tag="train_progma_${decay}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene tanks_and_temples/train --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --res-schedule "0.5:0,1.0:1500" \
    --cull-interval 1 --masked-adam $dflag \
    --out $OUT/$tag.json > $OUT/$tag.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}
run 0 0 0 &
run 1 1 0 &
run 2 2 0 &
run 3 0 1 &
run 4 1 1 &
run 5 2 1 &
wait
echo R65EXP1_W3_DONE
cat $OUT/rc.txt