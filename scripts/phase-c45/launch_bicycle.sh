#!/bin/bash
# Launch bicycle 30K + remaining 5K comparisons on free GPUs
cd ~/3dgs-renderer-benchmark

# GPU 3: D_baseline 30K bicycle
echo "Launching D_baseline 30K bicycle on GPU 3"
CUDA_VISIBLE_DEVICES=3 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_baseline --scene bicycle --iters 30000 > results/a100/phase-c45/log_D30_baseline_bicycle.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 4: D_sep_freq8 30K bicycle
echo "Launching D_sep_freq8 30K bicycle on GPU 4"
CUDA_VISIBLE_DEVICES=4 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_sep_freq8 --scene bicycle --iters 30000 > results/a100/phase-c45/log_D30_sep_freq8_bicycle.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 5: D_sep_ssim 5K room (separable SSIM only, no freq8, for comparison)
echo "Launching D_sep_ssim 5K room on GPU 5"
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track D --config D_sep_ssim --scene room --iters 5000 > results/a100/phase-c45/log_D_sep_ssim.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

# GPU 7: B3_psnr_aware 5K room (we haven't run this yet)
echo "Launching B3_psnr_aware 5K room on GPU 7"
CUDA_VISIBLE_DEVICES=7 nohup python3 -u scripts/phase-c45/c45_unified_experiment.py --track B --config B3_psnr_aware --iters 5000 > results/a100/phase-c45/log_B3.txt 2>&1 < /dev/null &
echo "  PID=$!"
sleep 3

echo "All launched."
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
