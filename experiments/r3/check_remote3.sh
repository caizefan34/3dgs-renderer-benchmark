#!/bin/bash
set -u
REPO="$(ls -d /home/liaoyuanjun/3dgs-renderer-benchmark 2>/dev/null)"
LOG="$REPO/results/reference_v1/r3/5000/runner.log"
echo "=== tail -20 ==="
tail -20 "$LOG" 2>&1
echo "=== iter lines ==="
grep -E "iter" "$LOG" 2>&1
echo "=== file size ==="
stat -c "%s bytes  %y" "$LOG" 2>&1
