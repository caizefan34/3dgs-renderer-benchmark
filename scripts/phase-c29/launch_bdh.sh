#!/bin/bash
cd /home/liaoyuanjun/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=5 nohup python3 scripts/phase-c29/b_param_frequency.py --out results/phase-c29/b_param_frequency.json --n_steps 300 > results/phase-c29/b.log 2>&1 &
echo B GPU5
CUDA_VISIBLE_DEVICES=6 nohup python3 scripts/phase-c29/d_densification_frequency.py --out results/phase-c29/d_densification_frequency.json --n_steps 300 > results/phase-c29/d.log 2>&1 &
echo D GPU6
CUDA_VISIBLE_DEVICES=7 nohup python3 scripts/phase-c29/h_event_prediction.py --out results/phase-c29/h_event_prediction.json --n_steps 500 > results/phase-c29/h.log 2>&1 &
echo H GPU7
