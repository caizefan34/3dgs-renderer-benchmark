#!/bin/bash
# R3 status check v3
REPO=$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)
R3DIR="$REPO/results/reference_v1/r3"
LOG="$R3DIR/5000/runner.log"
echo "=== now ==="
date '+%H:%M:%S'
echo "=== iter count ==="
grep -c "iter 50" "$LOG" 2>/dev/null || echo 0
echo "=== last iter lines ==="
grep "iter 50" "$LOG" 2>/dev/null | tail -3
echo "=== tail ==="
tail -3 "$LOG" 2>/dev/null
echo "=== marker ==="
ls -la "$R3DIR/COMPLETE.marker" 2>/dev/null || echo "not yet"
echo "=== proc ==="
ps -o pid,etime,time,pcpu -p 328374 2>/dev/null | tail -1
