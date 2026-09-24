#!/bin/bash
# Launch all 5 Track B ablation experiments in parallel
cd ~/3dgs-renderer-benchmark

for scale in 1.0 0.875 0.75 0.625 0.5; do
    gpu=$(($(echo "1.0 0.875 0.75 0.625 0.5" | tr ' ' '\n' | grep -n "^${scale}$" | cut -d: -f1) - 1))
    scale_str=$(echo $scale | tr -d '.')
    echo "Launching scale=$scale on GPU=$gpu -> track_b_log_${scale_str}.txt"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u scripts/phase-c42/track_b_ablation.py --scale $scale > results/a100/phase-c42/track_b_log_${scale_str}.txt 2>&1 &
    echo "  PID=$!"
    sleep 2
done

echo "All launched. Checking processes..."
sleep 5
ps aux | grep track_b_ablation | grep -v grep
