#!/bin/bash
# Minimal R3 status check - glob path
REPO=$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)
echo "REPO=$REPO"
echo "=== date ==="
date '+%H:%M:%S'
echo "=== r3 dirs ==="
ls -la "$REPO/results/reference_v1/r3/" 2>&1
echo "=== find logs ==="
find "$REPO/results/reference_v1/r3" -name "*.log" -o -name "*.json" 2>/dev/null | head -10
LOG=$(find "$REPO/results/reference_v1/r3" -name "runner.log" 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  echo "--- tail ---"
  tail -3 "$LOG"
fi
echo "=== procs ==="
ps aux | grep "[r]3_certificate" | awk '{print "PID="$2, "CPU="$3, "MEM="$4, "STAT="$8, "TIME="$10}'
