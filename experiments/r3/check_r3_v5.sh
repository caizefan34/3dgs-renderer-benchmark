#!/bin/bash
# R3 status check - file size + progress only
REPO=$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)
LOG="$REPO/results/reference_v1/r3/5000/runner.log"
date '+%H:%M:%S'
stat -c "%s bytes  %y" "$LOG" 2>/dev/null
grep -c "iter 50" "$LOG" 2>/dev/null || echo "no iter lines"
tail -1 "$LOG" 2>/dev/null
