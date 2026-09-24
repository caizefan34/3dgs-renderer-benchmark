#!/bin/bash
# Launch all Phase C44 experiments in parallel
cd ~/3dgs-renderer-benchmark

# Kill any leftover processes
pkill -f c44_unified 2>/dev/null
sleep 2

mkdir -p results/a100/phase-c44

# GPU 0: Baseline (scale=1.0, SSIM every iter)
echo "Launching baseline on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup python3 -u scripts/phase-c44/c44_unified_experiment.py --track baseline --config baseline > results/a100/phase-c44/log_baseline.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 1: A1 (0.5 first 1000, then 0.75)
echo "Launching A1 on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup python3 -u scripts/phase-c44/c44_unified_experiment.py --track A --config A1 > results/a100/phase-c44/log_A1.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 2: A2 (0.5 first 2000, then 0.75)
echo "Launching A2 on GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup python3 -u scripts/phase-c44/c44_unified_experiment.py --track A --config A2 > results/a100/phase-c44/log_A2.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 3: A3 (0.75 first 3000, then 1.0)
echo "Launching A3 on GPU 3"
CUDA_VISIBLE_DEVICES=3 nohup python3 -u scripts/phase-c44/c44_unified_experiment.py --track A --config A3 > results/a100/phase-c44/log_A3.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 4: C freq2 (SSIM every 2 iters)
echo "Launching C-freq2 on GPU 4"
CUDA_VISIBLE_DEVICES=4 nohup python3 -u scripts/phase-c44/c44_unified_experiment.py --track C --config freq2 > results/a100/phase-c44/log_C_freq2.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 5: C freq4 (SSIM every 4 iters)
echo "Launching C-freq4 on GPU 5"
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/phase-c44/c44_unified_experiment.py --track C --config freq4 > results/a100/phase-c44/log_C_freq4.txt 2>&1 &
echo "  PID=$!"
sleep 3

# GPU 6: C freq8 (SSIM every 8 iters)
echo "Launching C-freq8 on GPU 6"
CUDA_VISIBLE_DEVICES=6 nohup python3 -u scripts/phase-c44/c44_unified_experiment.py --track C --config freq8 > results/a100/phase-c44/log_C_freq8.txt 2>&1 &
echo "  PID=$!"
sleep 3

echo "All launched. Checking..."
sleep 5
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
