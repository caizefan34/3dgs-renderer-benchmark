#!/bin/bash
# R3 detailed check v5
REPO=$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)
R3DIR="$REPO/results/reference_v1/r3"
LOG="$R3DIR/5000/runner.log"
echo "=== time ==="
date '+%H:%M:%S'
ps -o pid,etime,time,pcpu,stat -p 328374 2>/dev/null
echo "=== log amd last 3 ==="
ls -la "$R3DIR/5000/" 2>/dev/null
tail -3 "$LOG" 2>/dev/null
