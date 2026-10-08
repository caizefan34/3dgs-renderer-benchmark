#!/bin/bash
echo "===== ROOM ====="
tail -30 /root/epic05_n1d/logs/n1d_room.log 2>/dev/null
echo "===== BICYCLE ====="
tail -30 /root/epic05_n1d/logs/n1d_bicycle.log 2>/dev/null
echo "===== GPU ====="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
