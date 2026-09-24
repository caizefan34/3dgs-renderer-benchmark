#!/bin/bash
# Comprehensive state check
R3DIR="$(ls -d /home/*/3dgs-renderer-benchmark/results/reference_v1/r3 2>/dev/null | head -1)"
echo "=== all dirs ==="
ls -la "$R3DIR" 2>&1
echo "=== 5000 log tail ==="
tail -5 "$R3DIR/5000/runner.log" 2>&1
echo "=== processes ==="
ps -eo pid,ppid,etime,time,pcpu,stat,args --sort=-pcpu 2>/dev/null | grep -E "python|r3" | grep -v grep | head -5
echo "=== find runner logs ==="
find /home/liaoyu*/3dgs-renderer-benchmark -name "runner.log" -exec echo {} \; -exec tail -1 {} \; 2>/dev/null
