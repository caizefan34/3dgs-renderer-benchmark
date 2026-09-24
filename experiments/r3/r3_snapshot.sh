#!/bin/bash
# R3 status snapshot - compact one-shot report
REPO="$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)"
R3DIR="$REPO/results/reference_v1/r3"
echo "TIME=$(date '+%H:%M:%S')"
echo "PROC_ALIVE=$(pgrep -fc r3_certificate_runner || echo 0)"
echo "LOGSIZE=$(stat -c '%s' "$R3DIR/5000/runner.log" 2>/dev/null || echo 0)"
echo "ITER_LINES=$(grep -c 'iter 50' "$R3DIR/5000/runner.log" 2>/dev/null || echo 0)"
echo "LAST_LINE=$(tail -1 "$R3DIR/5000/runner.log" 2>/dev/null)"
echo "JSON_COUNT=$(find "$R3DIR" -name '*.json' 2>/dev/null | wc -l)"
echo "NPZ_COUNT=$(find "$R3DIR" -name '*.npz' 2>/dev/null | wc -l)"
echo "MARKER=$(ls "$R3DIR/COMPLETE.marker" 2>/dev/null || echo not_yet)"
echo "CPU_PCT=$(top -b -n1 -p $(pgrep -f r3_certificate_runner | head -1) 2>/dev/null | tail -1 | awk '{print $9}')"
