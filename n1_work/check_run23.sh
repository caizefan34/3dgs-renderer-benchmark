#!/bin/bash
ps aux | grep -E "run23\.sh|n1_pass_grads|n1_pass_actions|n1_stage3" | grep -v grep
echo "=== run23.log ==="
cat /root/epic05_n1/logs/run23.log 2>/dev/null
echo "=== room ==="
tail -5 /root/epic05_n1/logs/stage23_room.log 2>/dev/null
echo "=== bicycle ==="
tail -5 /root/epic05_n1/logs/stage23_bicycle.log 2>/dev/null
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
