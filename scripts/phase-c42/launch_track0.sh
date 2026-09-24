#!/bin/bash
# Launch all 5 Track 0 (continuous pruning) experiments in parallel
cd ~/3dgs-renderer-benchmark

# Kill any leftover processes
pkill -f track_b_ablation 2>/dev/null
sleep 2

for scale in 1.0 0.875 0.75 0.625 0.5; do
    gpu=$(($(echo "1.0 0.875 0.75 0.625 0.5" | tr ' ' '\n' | grep -n "^${scale}$" | cut -d: -f1) - 1))
    scale_str=$(echo $scale | tr -d '.')
    echo "Launching scale=$scale on GPU=$gpu -> track0_pruned_${scale_str}.txt"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u scripts/phase-c42/track0_pruned_ablation.py --scale $scale > results/a100/phase-c42/track0_log_${scale_str}.txt 2>&1 &
    echo "  PID=$!"
    sleep 3
done

echo "All launched. Checking..."
sleep 5
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
