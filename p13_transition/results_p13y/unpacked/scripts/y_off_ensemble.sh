#!/bin/bash
# P13-Y supplementary: capture-OFF ensemble (400 iters) to characterize the
# intrinsic cross-process divergence distribution (3 additional OFF replays).
set -u
cd /root/epic05_p13y
PY=/root/miniforge3/envs/gsplat/bin/python
TR=publication_trainer_p13y_capture.py
COMMON="--arm c0 --scene room --iterations 400 --gpu 2 --lpips --final-eval all --timing-grade PUBLICATION --eps2d 0.3 --loss fused"
for K in 3 4 5; do
  R=runs/y_neutral_s_room_off$K
  bash p13y_unit.sh y_neutral_s_off$K logs/y_neutral_s_off$K.log -- $PY $TR $COMMON \
    --outdir $R --p13y-cam-log $R/cam_log.json --p13y-iter-log $R/p13y_iter_log.json
  echo "SOFF${K}_RC=$?"
done
echo OFF_ENSEMBLE_DONE
