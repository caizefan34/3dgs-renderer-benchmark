#!/bin/bash
cd /root/epic05_p11
ls -la logs/ dumps/
echo "=== procs ==="
ps aux | grep -E "p11_gate|p11_resume|p11_run" | grep -v grep
echo "=== gpu0 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -1
