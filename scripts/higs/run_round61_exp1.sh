#!/bin/bash
# Round-61 exp1: isolate the union-mask-prune quality collapse mechanism on
# garden 720p 120-step smoke.
#   GPU0: fresh eval mask at every densify step (no opacity gate)
#   GPU1: opacity gate 0.1 (stale eval mask, eval-step refresh only)
# Controls: r60 masked-adam-only PSNR 22.49; r61 naive PSNR 20.66.
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR61exp1
mkdir -p $OUT

common="--base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts --n-train 4 --n-eval 3 --steps 120 --width 1280 --height 720 --seed 0 --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 --error-refresh-every 25 --error-lambda 0.7 --eval-every 60 --lr-decay 0.1 --densify-window 1500 --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --anchor-densify --anchor-densify-every 2 --cull-interval 1 --masked-adam"
(CUDA_VISIBLE_DEVICES=0 $PY $BENCH $common --mask-prune --mask-prune-eval-refresh 5 --out $OUT/exp_fresh.json > $OUT/exp_fresh.log 2>&1; echo FRESH_RC=$? > $OUT/fresh.rc) &
(CUDA_VISIBLE_DEVICES=1 $PY $BENCH $common --mask-prune --mask-prune-opacity 0.1 --out $OUT/exp_opac.json > $OUT/exp_opac.log 2>&1; echo OPAC_RC=$? > $OUT/opac.rc) &
wait
echo EXP1_DONE
cat $OUT/fresh.rc $OUT/opac.rc