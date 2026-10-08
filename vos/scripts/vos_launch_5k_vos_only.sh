#!/bin/bash
# Relaunch VOS 5K only (C0_5K completed).
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
cd "$D"
setsid nohup $PY vos_train_driver.py --scene room --iterations 5000 --gpu 2 \
  --vos-start 1000 --final-eval subset \
  --outdir "$D"/vos_runs/vos_room_5k < /dev/null > "$D"/logs/vos_vos_room_5k.log 2>&1 &
echo "VOS_5K pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
