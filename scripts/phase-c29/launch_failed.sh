#!/bin/bash
cd ~/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/phase-c29/b_param_frequency.py --out results/phase-c29/b_param_frequency.json --n_steps 300 > results/phase-c29/b.log 2>&1 &
echo B GPU0
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/phase-c29/d_densification_frequency.py --out results/phase-c29/d_densification_frequency.json --n_steps 300 > results/phase-c29/d.log 2>&1 &
echo D GPU1
CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/phase-c29/f_camera_utility.py --out results/phase-c29/f_camera_utility.json --n_cameras 20 > results/phase-c29/f.log 2>&1 &
echo F GPU2
CUDA_VISIBLE_DEVICES=3 nohup python3 scripts/phase-c29/g_true_cross_iteration.py --out results/phase-c29/g_true_cross_iteration.json --n_iters 150 > results/phase-c29/g.log 2>&1 &
echo G GPU3
CUDA_VISIBLE_DEVICES=4 nohup python3 scripts/phase-c29/k_cuda_graph.py --out results/phase-c29/k_cuda_graph.json --n_iters 50 > results/phase-c29/k.log 2>&1 &
echo K GPU4
