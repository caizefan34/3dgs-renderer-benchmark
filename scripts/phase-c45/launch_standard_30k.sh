#!/bin/bash
# Launch 30K standard pruning validation on free GPUs (5, 6, 7)
cd ~/3dgs-renderer-benchmark

# GPU 5: D_baseline 30K room (standard pruning)
echo "Launching D_baseline 30K room (standard) on GPU 5"
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_baseline --scene room --iters 30000 --prune_config standard --save_suffix std30k > results/a100/phase-c45/log_D30std_baseline_room.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 6: D_sep_freq8 30K room (standard pruning)
echo "Launching D_sep_freq8 30K room (standard) on GPU 6"
CUDA_VISIBLE_DEVICES=6 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_sep_freq8 --scene room --iters 30000 --prune_config standard --save_suffix std30k > results/a100/phase-c45/log_D30std_sep_freq8_room.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 7: D_sep_ssim 30K room (standard pruning, sep SSIM only no freq8)
echo "Launching D_sep_ssim 30K room (standard) on GPU 7"
CUDA_VISIBLE_DEVICES=7 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_sep_ssim --scene room --iters 30000 --prune_config standard --save_suffix std30k > results/a100/phase-c45/log_D30std_sep_ssim_room.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

echo "All standard-pruning 30K experiments launched."
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
