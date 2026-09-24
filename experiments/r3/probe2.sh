#!/bin/bash
R3DIR="/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_1/r3"
echo "=== DIRS ==="
find "$R3DIR" -maxdepth 1 -type d 2>&1 | sort
echo "=== 5000 files ==="
ls -la "$R3DIR/5000/" 2>&1
echo "=== full_run.log tail ==="
tail -20 "$R3DIR/full_run.log" 2>&1