#!/bin/bash
set -u
REPO="$(ls -d /home/liaoyuanjun/3dgs-renderer-benchmark 2>/dev/null || echo /home/*/3dgs-renderer-benchmark)"
echo "REPO=$REPO"
echo "=== now ==="
date
echo "=== r3 root ==="
ls -la "$REPO/results/reference_v1/r3/" 2>&1
echo "=== r3 5000 dir ==="
ls -la "$REPO/results/reference_v1/r3/5000/" 2>&1
echo "=== runner log tail ==="
tail -6 "$REPO/results/reference_v1/r3/5000/runner.log" 2>&1
echo "=== iter lines count ==="
grep -c "iter 50" "$REPO/results/reference_v1/r3/5000/runner.log" 2>&1 || echo "no iter lines"
echo "=== procs ==="
ps aux | grep r3_certificate | grep -v grep | awk '{print $2, $3, $8, $11}'
