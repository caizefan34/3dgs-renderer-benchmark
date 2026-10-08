#!/bin/bash
# Isolation pair: 599 iterations, VOS from 101. NO densify events (they start
# at 600), NO eval fences; only the final fence before ckpt save. If the two
# ckpts are bitwise identical, the VOS step math + eager-foreach composition
# are exact on this trajectory and the 5K divergence is event-path related.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
R=$D/vos_runs
cd "$D"
setsid nohup $PY vos_train_driver.py --scene room --iterations 599 --gpu 1 \
  --vos-start 99999 --final-eval subset \
  --outdir "$R"/c0_room_599 < /dev/null > "$D"/logs/vos_c0_room_599.log 2>&1 &
echo "C0_599 pid=$!"
setsid nohup $PY vos_train_driver.py --scene room --iterations 599 --gpu 2 \
  --vos-start 100 --final-eval subset \
  --outdir "$R"/vos_room_599 < /dev/null > "$D"/logs/vos_vos_room_599.log 2>&1 &
echo "VOS_599 pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
