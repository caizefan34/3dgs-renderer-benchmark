#!/bin/bash
cd /root/epic05_p11
echo "=== pipeline.log ==="
cat logs/pipeline.log
echo "=== logs dir ==="
ls -la logs/
echo "=== running procs ==="
ps aux | grep -E "p11_" | grep -v grep | head -5
echo "=== gpu ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -4
