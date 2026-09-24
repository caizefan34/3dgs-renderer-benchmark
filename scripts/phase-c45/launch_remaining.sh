#!/bin/bash
# Relaunch killed experiments + launch 30K Track D validation
cd ~/3dgs-renderer-benchmark

# GPU 3: B2 gradient-aware (5K, relaunch)
echo "Relaunching B2 on GPU 3"
CUDA_VISIBLE_DEVICES=3 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track B --config B2_gradient_aware --iters 5000 > results/a100/phase-c45/log_B2.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 7: D_baseline 5K (relaunch, for comparison)
echo "Relaunching D_baseline 5K on GPU 7"
CUDA_VISIBLE_DEVICES=7 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_baseline --scene room --iters 5000 > results/a100/phase-c45/log_D_baseline.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 5: D_sep_freq8 5K (best C44 method, room)
echo "Launching D_sep_freq8 5K on GPU 5"
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_sep_freq8 --scene room --iters 5000 > results/a100/phase-c45/log_D_sep_freq8.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 0: D_baseline 30K (room, full training)
echo "Launching D_baseline 30K room on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_baseline --scene room --iters 30000 > results/a100/phase-c45/log_D30_baseline_room.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 1: D_sep_freq8 30K (room, full training)
echo "Launching D_sep_freq8 30K room on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_sep_freq8 --scene room --iters 30000 > results/a100/phase-c45/log_D30_sep_freq8_room.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 2: D_baseline 30K (garden)
echo "Launching D_baseline 30K garden on GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_baseline --scene garden --iters 30000 > results/a100/phase-c45/log_D30_baseline_garden.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 6: D_sep_freq8 30K (garden)
echo "Launching D_sep_freq8 30K garden on GPU 6"
CUDA_VISIBLE_DEVICES=6 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_sep_freq8 --scene garden --iters 30000 > results/a100/phase-c45/log_D30_sep_freq8_garden.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

echo "All launched."
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
