#!/bin/bash
# Read full run log tail
R3DIR="$(ls -d /home/*/3dgs-renderer-benchmark/results/reference_1/r3 2>/dev/null || ls -d /home/*/3dgs-renderer-benchmark/results/reference_1/r3 2>/dev/null)"
LOG="$R3DIR/full_run.log"
echo "=== full_run.log tail -30 ==="
tail -30 "$LOG" 2>&1
echo "=== dirs ==="
find "$R3DIR" -maxdepth 1 -type d 2>/dev/null | sort
