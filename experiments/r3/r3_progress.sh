#!/bin/bash
# R3 progress with timing history
REPO="$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)"
R3DIR="$REPO/results/reference_v1/r3"
LOG="$R3DIR/5000/runner.log"
echo "TIME=$(date '+%H:%M:%S')"
echo "=== all iter summary lines ==="
grep -E "iter 50" "$LOG" 2>/dev/null
echo "=== last 10 lines ==="
tail -10 "$LOG" 2>/dev/null
echo "=== file mtimes ==="
stat -c "%y %s %n" "$R3DIR/5000/runner.log" 2>/dev/null
echo "=== total lines count ==="
wc -l "$LOG" 2>/dev/null
