#!/bin/bash
# Launch all Phase C45 experiments
cd ~/3dgs-renderer-benchmark
pkill -f c45_unified 2>/dev/null
pkill -f post_c44 2>/dev/null
sleep 2
mkdir -p results/a100/phase-c45

# GPU 0: Post-C44 profiling (quick, ~2 min)
echo "Launching post-C44 profile on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup python3 -u scripts/phase-c45/post_c44_profile.py > results/a100/phase-c45/log_profile.txt 2>&1 &
echo "  PID=$!"
sleep 5

# GPU 1: Track B1 early_mid_late (5K iters)
echo "Launching B1_early_mid_late on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track B --config B1_early_mid_late --iters 5000 > results/a100/phase-c45/log_B1_eml.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 2: Track B1 decreasing (5K iters)
echo "Launching B1_decreasing on GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track B --config B1_decreasing --iters 5000 > results/a100/phase-c45/log_B1_dec.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 3: Track B2 gradient-aware (5K iters)
echo "Launching B2_gradient_aware on GPU 3"
CUDA_VISIBLE_DEVICES=3 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track B --config B2_gradient_aware --iters 5000 > results/a100/phase-c45/log_B2.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 4: Track C1 Laplacian (5K iters)
echo "Launching C1_laplacian on GPU 4"
CUDA_VISIBLE_DEVICES=4 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track C --config C1_laplacian --iters 5000 > results/a100/phase-c45/log_C1.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 5: Track C2 FFT (5K iters)
echo "Launching C2_fft on GPU 5"
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track C --config C2_fft --iters 5000 > results/a100/phase-c45/log_C2.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 6: Track C3 edge-aware (5K iters)
echo "Launching C3_edge on GPU 6"
CUDA_VISIBLE_DEVICES=6 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track C --config C3_edge --iters 5000 > results/a100/phase-c45/log_C3.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 7: Track D baseline 5K (room, for quick comparison)
echo "Launching D_baseline on GPU 7"
CUDA_VISIBLE_DEVICES=7 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_baseline --scene room --iters 5000 > results/a100/phase-c45/log_D_baseline.txt 2>&1 &
echo "  PID=$!"
sleep 3

echo "All 8 experiments launched."
sleep 5
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
