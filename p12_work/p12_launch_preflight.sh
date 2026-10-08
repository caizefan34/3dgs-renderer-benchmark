#!/bin/bash
# P12 preflight launcher: 4 arms x 3 scenes x 220 iters on clean GPUs 0/1/2.
# Preflight is verification only -- NOTHING is tuned here.
set -u
mkdir -p /root/epic05_p12/logs /root/epic05_p12/artifacts
cd /root/epic05_p12
nohup /root/miniforge3/envs/gsplat/bin/python p12_preflight.py > logs/preflight.log 2>&1 &
echo "PREFLIGHT_PID=$!"
