#!/bin/bash
# P13-X main run: room 16K diagnostic run + contamination companion + X6 oracle
# Back-to-back on the SAME physical GPU (contamination policy).
cd /root/epic05_p13x
PY=/root/miniforge3/envs/gsplat/bin/python
export CUDA_VISIBLE_DEVICES=4
mkdir -p logs runs artifacts
echo "MAIN_RUN_START $(date -u +%FT%TZ)" > logs/main_run.log

COMMON="--arm c0 --scene room --loss fused --gpu 0 --eps2d 0.3 --iterations 16000 --final-eval subset --timing-grade FUNCTIONAL_ONLY --lpips --checkpoints --p13x-rng-footprint 1"

$PY publication_trainer_p13x.py $COMMON \
  --p13x-diag 1 --p13x-diag-start 15000 --p13x-diag-end 16000 \
  --p13x-diag-stride 250 --p13x-lags 250,500 --p13x-anchors 8 \
  --p13x-x4-samples 256 --p13x-x5 1 --p13x-x5-anchors 8 \
  --outdir runs/room_diag > logs/room_diag.log 2>&1
echo "DIAG_EXIT=$?" >> logs/main_run.log

$PY publication_trainer_p13x.py $COMMON \
  --p13x-diag 0 \
  --outdir runs/room_diagoff > logs/room_diagoff.log 2>&1
echo "DIAGOFF_EXIT=$?" >> logs/main_run.log

$PY p13x_x6_oracle.py --scene room --ckpt runs/room_diag/ckpt_16000.pt \
  --p13x-raw-dir runs/room_diag/p13x/raw --outdir artifacts/x6_room \
  --gpu 0 --reps 30 > logs/x6_room.log 2>&1
echo "X6_EXIT=$?" >> logs/main_run.log
echo "ALL_DONE $(date -u +%FT%TZ)" >> logs/main_run.log
