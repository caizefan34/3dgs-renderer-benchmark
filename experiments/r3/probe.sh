#!/bin/bash
R3DIR="$(ls -d /home/*/3dgs-renderer-benchmark/results/reference_1/r3 2>/dev/null | head -1)"
echo "=== TIME ==="
date "+%H:%M:%S"
echo "=== DIRS ==="
find "$R3DIR" -maxdepth 1 -type d | sort
echo "=== 5000 files ==="
ls -la "$R3DIR/5000/" 2>/dev/null
echo "=== full_run.log tail ==="
tail -20 "$R3DIR/full_run.log" 2>/dev/null
echo "=== proc ==="
ps -o pid,etime,time,pcpu,stat -p 328374 2>/dev/null | tail -1