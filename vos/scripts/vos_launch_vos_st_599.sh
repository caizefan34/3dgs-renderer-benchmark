#!/bin/bash
# Decisive VOS-exactness test: VOS with eager attrs on the single-tensor path
# (--eager-single-tensor) vs the pure A_s arm (--as-mode). Both optimizers are
# single-tensor-consistent; the ONLY difference is VOS's deferred shs path.
# If these two 599-iter ckpts are bitwise identical, the integration is exact
# and every observed divergence vs the stock foreach trainer is the
# torch-internal A_f-vs-A_s envelope (chaotically amplified), not VOS.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
cd "$D"
setsid nohup $PY vos_train_driver.py --scene room --iterations 599 --gpu 4 \
  --vos-start 100 --eager-single-tensor --final-eval subset \
  --outdir "$D"/vos_runs/vos_st_room_599 < /dev/null > "$D"/logs/vos_vos_st_room_599.log 2>&1 &
echo "VOS_ST_599 pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
