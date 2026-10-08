#!/bin/bash
# VOS integration diagnostics (both vs the pure A_s arm):
#  (a) CONTROL: eager-single-tensor, vos-start beyond the run -> the proxy and
#      all wrapper machinery active but ZERO VOS steps. Must be bitwise vs
#      as_room_599, else the wrapper itself perturbs training.
#  (b) SINGLE-STEP: VOS active only at t=599 (one VOS step + final fence).
#      If (a) passes but (b) fails, the single VOS step is wrong and can be
#      diffed directly.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
cd "$D"
setsid nohup $PY vos_train_driver.py --scene room --iterations 599 --gpu 5 \
  --vos-start 99999 --eager-single-tensor --final-eval subset \
  --outdir "$D"/vos_runs/ctl_st_room_599 < /dev/null > "$D"/logs/vos_ctl_st_room_599.log 2>&1 &
echo "CTL pid=$!"
setsid nohup $PY vos_train_driver.py --scene room --iterations 599 --gpu 6 \
  --vos-start 598 --eager-single-tensor --final-eval subset \
  --outdir "$D"/vos_runs/vos1_st_room_599 < /dev/null > "$D"/logs/vos_vos1_st_room_599.log 2>&1 &
echo "ONESTEP pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
