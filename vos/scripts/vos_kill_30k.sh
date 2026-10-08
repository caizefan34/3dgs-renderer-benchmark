#!/bin/bash
# Kill the mis-pinned 30K campaign (all contexts stacked on GPU 0) and
# verify GPUs are clean.
set -u
pkill -f vos_train_driver.py 2>/dev/null; sleep 2
pkill -9 -f vos_train_driver.py 2>/dev/null; sleep 3
for s in room bicycle garden; do
  for arm in c0 vos; do
    rm -rf "/root/vos_traces/vos_runs/${arm}_${s}_30k"
  done
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "CLEANED $(date -u +%FT%TZ)"
