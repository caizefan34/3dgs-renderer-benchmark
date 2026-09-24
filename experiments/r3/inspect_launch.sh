#!/bin/bash
# Inspect launch script and process tree
echo "=== launch script content ==="
cat /tmp/launch_r3_full.sh 2>/dev/null | head -40
echo "=== process tree ==="
ps -ef | grep -E "[r]3_certificate|run_" | grep -v grep | awk '{print $2, $3, $8, $9, $10, $11}'
echo "=== any logs ==="
find /home/liaoyunhan/3dgs-renderer-benchmark/results/reference_1/r3 -type f 2>/dev/null | head -10
