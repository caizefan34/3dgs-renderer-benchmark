#!/bin/bash
cd /root/epic05_p13x
/root/miniforge3/envs/gsplat/bin/python -m py_compile p13x_x6_oracle.py p13x_report.py && echo COMPILE_OK
nohup bash launch_main.sh > logs/launch_main_wrapper.log 2>&1 < /dev/null &
echo "MAIN_PID=$!"
