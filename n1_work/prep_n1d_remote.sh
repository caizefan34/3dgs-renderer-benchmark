#!/bin/bash
set -e
mkdir -p /root/epic05_n1d/artifacts /root/epic05_n1d/logs
cd /root/epic05_n1d
echo "remote staging OK"
ls -la /root/epic05_n1/artifacts/N1_PROBE_LIST.json /root/epic05_n1/n1_core.py
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
