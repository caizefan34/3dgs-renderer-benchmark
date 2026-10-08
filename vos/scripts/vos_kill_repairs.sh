#!/bin/bash
# Kill my compromised re-pair chain; poll until MR0 clears.
pkill -f vos_launch_clean_repairs 2>/dev/null
pkill -f vos_train_driver 2>/dev/null
sleep 3
rm -rf /root/vos_traces/vos_runs/vos_bicycle_30k_b
echo "=== after kill ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "=== MR0 still running? ==="
ps aux 2>/dev/null | grep -E "mr0_driver" | grep -v grep | head -2 || echo "MR0 done"
