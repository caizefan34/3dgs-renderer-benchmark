#!/bin/bash
# Check progression into next windows
REPO="$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)"
R3DIR="$REPO/results/reference_v1/r3"
echo "=== R3 tree ==="
find "$R3DIR" -maxdepth 2 -type d 2>/dev/null | sort
echo "=== 15000 window dir ==="
ls -la "$R3DIR/15000/" 2>&1 | head -15
echo "=== 30000 window dir ==="
ls -la "$R3DIR/30000/" 2>&1 | head -15
echo "=== process ==="
ps -o pid,etime,time,pcpu,stat -p 328374 2>/dev/null | tail -1
