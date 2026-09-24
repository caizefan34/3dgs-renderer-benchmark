#!/bin/bash
# Remote R3 status check
LOG=/home/liaoyunhan/3gds-renderer-benchmark/results/reference_v1/r3/5000/runner.log 2>/dev/null || true
ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null
ls -la /home/*/3dgs-renderer-benchmark/results/reference_v1/r3/ 2>/dev/null
echo "=== log stats ==="
stat -c "%s bytes  %y" /home/*/3dgs-renderer-benchmark/results/reference_v1/r3/5000/runner.log 2>/dev/null
echo "=== last 3 lines ==="
tail -3 /home/*/3dgs-renderer-benchmark/results/reference_v1/r3/5000/runner.log 2>/dev/null
date
