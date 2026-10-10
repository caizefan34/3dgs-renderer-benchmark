#!/bin/bash
set +e
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
echo "=== best free GPU ==="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -2
echo "=== room dataset present? ==="
ls -la $REPO/../benchmark_data/mipnerf360/room/sparse/0/ 2>/dev/null | head
ls /home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage5/ 2>/dev/null | head -3
echo "=== stage harness to repo ==="
cp /tmp/C51J_harness.py $REPO/C51J_harness.py && echo "staged to $REPO"
echo "DONE"
