#!/bin/bash
cd ~/3dgs-renderer-benchmark
mkdir -p results/phase-c29
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/phase-c29/a_step_value.py --out results/phase-c29/a_step_value.json --n_steps 500 > results/phase-c29/a.log 2>&1 &
echo "A GPU0"
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/phase-c29/b_param_frequency.py --out results/phase-c29/b_param_frequency.json --n_steps 300 > results/phase-c29/b.log 2>&1 &
echo "B GPU1"
CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/phase-c29/c_param_compute.py --out results/phase-c29/c_param_compute.json > results/phase-c29/c.log 2>&1 &
echo "C GPU2"
CUDA_VISIBLE_DEVICES=3 nohup python3 scripts/phase-c29/e_camera_importance.py --out results/phase-c29/e_camera_importance.json --n_cameras 15 > results/phase-c29/e.log 2>&1 &
echo "E GPU3"
CUDA_VISIBLE_DEVICES=4 nohup python3 scripts/phase-c29/f_camera_utility.py --out results/phase-c29/f_camera_utility.json --n_cameras 20 > results/phase-c29/f.log 2>&1 &
echo "F GPU4"
CUDA_VISIBLE_DEVICES=5 nohup python3 scripts/phase-c29/g_true_cross_iteration.py --out results/phase-c29/g_true_cross_iteration.json --n_iters 150 > results/phase-c29/g.log 2>&1 &
echo "G GPU5"
CUDA_VISIBLE_DEVICES=6 nohup python3 scripts/phase-c29/j_communication.py --out results/phase-c29/j_communication.json > results/phase-c29/j.log 2>&1 && echo "J GPU6" || echo "J FAIL"
CUDA_VISIBLE_DEVICES=6 nohup python3 scripts/phase-c29/k_cuda_graph.py --out results/phase-c29/k_cuda_graph.json --n_iters 50 > results/phase-c29/k.log 2>&1 && echo "K GPU6" || echo "K FAIL"
CUDA_VISIBLE_DEVICES=6 nohup python3 scripts/phase-c29/l_state_reuse.py --out results/phase-c29/l_state_reuse.json --n_repeats 50 > results/phase-c29/l.log 2>&1 &
echo "L GPU6 bg"
CUDA_VISIBLE_DEVICES=7 nohup python3 scripts/phase-c29/d_densification_frequency.py --out results/phase-c29/d_densification_frequency.json --n_steps 300 > results/phase-c29/d.log 2>&1 && echo "D GPU7" || echo "D FAIL"
CUDA_VISIBLE_DEVICES=7 nohup python3 scripts/phase-c29/h_event_prediction.py --out results/phase-c29/h_event_prediction.json --n_steps 500 > results/phase-c29/h.log 2>&1 && echo "H GPU7" || echo "H FAIL"
CUDA_VISIBLE_DEVICES=7 nohup python3 scripts/phase-c29/i_multigpu_sharding.py --out results/phase-c29/i_multigpu_sharding.json --n_cameras 100 > results/phase-c29/i.log 2>&1 &
echo "I GPU7 bg"
echo "ALL LAUNCHED"
