#!/bin/bash
# R3 status check - simple output
echo "=== date ==="
date '+%H:%M:%S'
echo "=== processes ==="
ps aux | grep "[r]3_certificate" | awk '{print $2, $3, $4, $5, $6, $8, $11}'
echo "=== r3 root ==="
ls -la /home/*/3dgs-renderer-benchmark/results/reference_v1/r3/ 2>&1
echo "=== 5000 dir ==="
ls -la /home/*/3dgs-renderer-benchmark/results/reference_v1/r3/5000/ 2>&1
echo "=== 5000 log tail ==="
tail -3 /home/*/3dgs-renderer-benchmark/results/reference_v1/r3/5000/runner.log 2>&1
echo "=== count iter lines ==="
grep -c "iter 50" /home/*/3dgs-renderer-benchmark/results/reference_v1/r3/5000/runner.log 2>&1
