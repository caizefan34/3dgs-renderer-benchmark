#!/bin/bash
# P13-X bicycle chain: 16K diag-on + diag-off companion + X6 (GPU 5, back-to-back)
cd /root/epic05_p13x
PY=/root/miniforge3/envs/gsplat/bin/python
export CUDA_VISIBLE_DEVICES=5
mkdir -p logs runs artifacts
echo "BICYCLE_START $(date -u +%FT%TZ)" > logs/bicycle.log

COMMON="--arm c0 --scene bicycle --loss fused --gpu 0 --eps2d 0.3 --iterations 16000 --final-eval subset --timing-grade FUNCTIONAL_ONLY --lpips --checkpoints --p13x-rng-footprint 1"

$PY publication_trainer_p13x.py $COMMON \
  --p13x-diag 1 --p13x-diag-start 15000 --p13x-diag-end 16000 \
  --p13x-diag-stride 250 --p13x-lags 250,500 --p13x-anchors 8 \
  --p13x-x4-samples 256 --p13x-x5 1 --p13x-x5-anchors 8 \
  --outdir runs/bicycle_diag > logs/bicycle_diag.log 2>&1
echo "DIAG_EXIT=$?" >> logs/bicycle.log

$PY publication_trainer_p13x.py $COMMON \
  --p13x-diag 0 \
  --outdir runs/bicycle_diagoff > logs/bicycle_diagoff.log 2>&1
echo "DIAGOFF_EXIT=$?" >> logs/bicycle.log

$PY p13x_x6_oracle.py --scene bicycle --ckpt runs/bicycle_diag/ckpt_16000.pt \
  --p13x-raw-dir runs/bicycle_diag/p13x/raw --outdir artifacts/x6_bicycle \
  --gpu 0 --reps 30 > logs/x6_bicycle.log 2>&1
echo "X6_EXIT=$?" >> logs/bicycle.log
echo "ALL_DONE $(date -u +%FT%TZ)" >> logs/bicycle.log
