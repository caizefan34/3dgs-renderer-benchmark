#!/bin/bash
# Round-61 exp5: LPIPS canonical work size on garden 720p 120-step smoke.
#   GPU0: --lpips-work-size 0 (control, = round-60 baseline config)
#   GPU1: --lpips-work-size 256
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR61exp5
mkdir -p $OUT

common="--base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 120 --width 1280 --height 720 --seed 0 --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 --error-refresh-every 25 --error-lambda 0.7 --eval-every 60 --lr-decay 0.1 --densify-window 1500 --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --anchor-densify --anchor-densify-every 2 --cull-interval 1 --masked-adam"
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --out $OUT/exp_ws0.json > $OUT/exp_ws0.log 2>&1; echo WS0_RC=$? > $OUT/ws0.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --lpips-work-size 256 --out $OUT/exp_ws256.json > $OUT/exp_ws256.log 2>&1; echo WS256_RC=$? > $OUT/ws256.rc) &
wait
echo EXP5_WAVE1_DONE
cat $OUT/ws0.rc $OUT/ws256.rc
# Wave 2
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --lpips-work-size 192 --out $OUT/exp_ws192.json > $OUT/exp_ws192.log 2>&1; echo WS192_RC=$? > $OUT/ws192.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --lpips-work-size 320 --out $OUT/exp_ws320.json > $OUT/exp_ws320.log 2>&1; echo WS320_RC=$? > $OUT/ws320.rc) &
wait
echo EXP5_DONE
cat $OUT/ws192.rc $OUT/ws320.rc