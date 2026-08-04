#!/bin/bash
# Round-61 smoke: garden 720p 120-step union-mask prune (masked-adam + mask-prune)
# vs round-60 masked-adam-only control.
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR61smoke
mkdir -p $OUT

common="--base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 120 --width 1280 --height 720 --seed 0 --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 --error-refresh-every 25 --error-lambda 0.7 --eval-every 60 --lr-decay 0.1 --densify-window 1500 --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --anchor-densify --anchor-densify-every 2 --cull-interval 1"
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --masked-adam --mask-prune --out $OUT/smoke_r61.json > $OUT/smoke_r61.log 2>&1; echo R61_RC=$? > $OUT/r61.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --masked-adam --out $OUT/smoke_r60.json > $OUT/smoke_r60.log 2>&1; echo R60_RC=$? > $OUT/r60.rc) &
wait
echo SMOKE_DONE
cat $OUT/r61.rc $OUT/r60.rc