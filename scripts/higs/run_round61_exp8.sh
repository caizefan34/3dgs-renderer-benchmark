#!/bin/bash
# Round-61 exp8: K4 x masked-Adam stack screen - 3 scenes x seed 0, 3000 steps.
# r59 K4 alone was +2% train_ms but bicycle PSNR -0.34; r60 masked-Adam (k1) was
# -34/-41% train_ms quality-positive. Stack test: --cull-interval 4 --masked-adam.
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR61exp8
mkdir -p $OUT

run () {
  local gpu=$1 scene=$2
  local extra=""
  if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  local sname=$(basename "$scene")
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1280 --height 720 --seed 0 \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval 4 --masked-adam \
    $extra --out $OUT/${sname}_k4ma_s0.json > $OUT/${sname}_k4ma_s0.log 2>&1
  echo "DONE $sname rc=$?" >> $OUT/rc.txt
}
run 0 mipnerf360/garden &
run 1 tanks_and_temples/train &
run 2 mipnerf360/bicycle &
wait
echo EXP8_DONE
cat $OUT/rc.txt