#!/bin/bash
# Launch the P11 pipeline detached with correct logging.
mkdir -p /root/epic05_p11/logs /root/epic05_p11/dumps /root/epic05_p11/artifacts
cd /root/epic05_p11
nohup bash p11_run_all.sh all > logs/pipeline.log 2>&1 &
echo "PIPELINE_PID=$!"
