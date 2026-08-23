#!/bin/bash
# Round-61 exp6: decisive LPIPS work-size test - 600-step, 3 scenes x {0,256}, seed 0.
#   GPU0-1 garden ws0/ws256, GPU2-3 train ws0/ws256, GPU4-5 bicycle ws0/ws256.
# Gate decision: if WS256 keeps >=1.5% train_ms win with PSNR/LPIPS within noise,
# proceed to exp7 (full 3000-step 3-seed sweep vs round-60 baselines).
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR61exp6
mkdir -p $OUT

run () {
  local gpu=$1 scene=$2 ws=$3
  local extra=""
  if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  local wsarg=""
  if [ "$ws" != "0" ]; then wsarg="--lpips-work-size $ws"; fi
  local sname=$(basename "$scene")
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 600 --width 1280 --height 720 --seed 0 \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 60 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval 1 --masked-adam \
    $extra $wsarg --out $OUT/${sname}_ws${ws}.json > $OUT/${sname}_ws${ws}.log 2>&1
  echo "DONE $sname ws$ws rc=$?" >> $OUT/rc.txt
}
run 0 mipnerf360/garden 0 &
run 1 mipnerf360/garden 256 &
run 2 tanks_and_temples/train 0 &
run 3 tanks_and_temples/train 256 &
run 4 mipnerf360/bicycle 0 &
run 5 mipnerf360/bicycle 256 &
wait
echo EXP6_DONE
cat $OUT/rc.txt