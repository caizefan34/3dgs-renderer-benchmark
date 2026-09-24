#!/bin/bash
echo "TIME=$(date '+%H:%M:%S')"
echo "--- 15000 runner log tail ---"
tail -6 /home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_1/r3/15000/runner.log 2>&1
echo "--- 15000 dir listing ---"
ls -la /home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_1/r3/15000/ 2>&1
echo "--- all R3 dirs ---"
find /home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_1/r3 -maxdepth 1 -type d 2>&1 | sort
