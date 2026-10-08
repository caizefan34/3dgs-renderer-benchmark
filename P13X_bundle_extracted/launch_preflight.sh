#!/bin/bash
cd /root/epic05_p13x
export P13X_PREFLIGHT_GPU=2
nohup /root/miniforge3/envs/gsplat/bin/python p13x_preflight.py > logs/preflight.log 2>&1 < /dev/null &
echo "PREFLIGHT_PID=$!"
