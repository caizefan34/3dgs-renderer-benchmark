#!/bin/bash
# P13-Y Sec.4 capture neutrality trio (room, 3500 iters, GPU 2).
# OFF1 vs ON vs OFF2: identical command, only the capture side channel differs.
set -u
cd /root/epic05_p13y
PY=/root/miniforge3/envs/gsplat/bin/python
TR=publication_trainer_p13y_capture.py
COMMON="--arm c0 --scene room --iterations 3500 --gpu 2 --lpips --final-eval all --timing-grade PUBLICATION --eps2d 0.3 --loss fused"

R1=runs/y_neutral_room_off1
R2=runs/y_neutral_room_on
R3=runs/y_neutral_room_off2

bash p13y_unit.sh y_neutral_off1 logs/y_neutral_off1.log -- $PY $TR $COMMON \
  --outdir $R1 --p13y-cam-log $R1/cam_log.json --p13y-iter-log $R1/p13y_iter_log.json
echo "OFF1_RC=$?"

bash p13y_unit.sh y_neutral_on logs/y_neutral_on.log -- $PY $TR $COMMON \
  --outdir $R2 --p13y-cam-log $R2/cam_log.json --p13y-iter-log $R2/p13y_iter_log.json \
  --p13y-capture-dir $R2/capture --p13y-parity-dir $R2/probes \
  --p13y-capture-iters 1000,3000,3400 --p13y-parity-iters 3000,3400 \
  --p13y-source-identity $R2/p13y_capture_identity.json
echo "ON_RC=$?"

bash p13y_unit.sh y_neutral_off2 logs/y_neutral_off2.log -- $PY $TR $COMMON \
  --outdir $R3 --p13y-cam-log $R3/cam_log.json --p13y-iter-log $R3/p13y_iter_log.json
echo "OFF2_RC=$?"
