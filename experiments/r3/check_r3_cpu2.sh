#!/bin/bash
# CPU sampling x2
ps -o pid,etime,time,pcpu,stat -p 328374 2>/dev/null
sleep 8
ps -o pid,etime,time,pcpu,stat -p 328374 2>/dev/null
echo ===nvidia===
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2
