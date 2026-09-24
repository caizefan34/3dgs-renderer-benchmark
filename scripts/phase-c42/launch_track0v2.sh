#!/bin/bash
cd ~/3dgs-renderer-benchmark

for scale in 1.0 0.875 0.75 0.625 0.5; do
    gpu=$(($(echo "1.0 0.875 0.75 0.625 0.5" | tr ' ' '\n' | grep -n "^${scale}$" | cut -d: -f1) - 1))
    scale_str=$(echo $scale | tr -d '.')
    echo "Launching scale=$scale on GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u scripts/phase-c42/track0v2_aggressive_pruning.py --scale $scale > results/a100/phase-c42/track0v2_log_${scale_str}.txt 2>&1 &
    echo "  PID=$!"
    sleep 3
done

echo "All launched."
sleep 5
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
