#!/bin/bash
# A_s reference arm: stock trainer with single-tensor Adam (foreach=False),
# NO VOS. 599 iterations, room, seed 42. Pairs with c0_room_599 (A_f) and
# vos_room_599 (VOS kernel == A_s semantics) to separate:
#   VOS-vs-A_s exactness  (should be bitwise)
#   A_f-vs-A_s envelope trajectory effect (torch-internal, not VOS's doing)
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
cd "$D"
setsid nohup $PY vos_train_driver.py --scene room --iterations 599 --gpu 3 \
  --as-mode --final-eval subset \
  --outdir "$D"/vos_runs/as_room_599 < /dev/null > "$D"/logs/vos_as_room_599.log 2>&1 &
echo "AS_599 pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
