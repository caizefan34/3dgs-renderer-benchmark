#!/bin/bash
# R3 detailed status v4
REPO=$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)
R3DIR="$REPO/results/reference_v1/r3"
LOG="$R3DIR/5000/runner.log"
echo "=== now ==="
date '+%H:%M:%S'
echo "=== proc CPU ==="
ps -o pid,etime,time,pcpu,stat,wchan:20 -p 328374 2>/dev/null
echo "=== log stat ==="
stat -c "%s bytes  modified %y" "$LOG" 2>/dev/null
echo "=== iter lines with numbering ==="
grep -n "iter 50" "$LOG" 2>/dev/null | tail -5
echo "=== very last 5 lines ==="
tail -5 "$LOG" 2>/dev/null
echo "=== count all lines ==="
wc -l "$LOG" 2>/dev/null
