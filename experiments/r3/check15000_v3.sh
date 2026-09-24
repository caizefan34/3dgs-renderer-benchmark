#!/bin/bash
R3DIR="/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/r3"
echo "TIME=$(date "+%H:%M:%S")"
echo "--- 15000 runner log tail ---"
tail -8 "$R3DIR/15000/runner.log" 2>&1
echo "--- iter count ---"
grep -c "iter" "$R3DIR/15000/runner.log" 2>/dev/null || echo 0
echo "--- 15000 files ---"
ls -la "$R3DIR/15000/" 2>&1
echo "--- disk ---"
df -h / | tail -1