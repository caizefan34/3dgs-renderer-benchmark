#!/bin/bash
# VOS Phase-8 functional gate: forced-early VOS 5K vs instrumented C0 5K.
# VOS active from iteration 1001 (INSIDE the densify region: exercises
# pre-densify fences, tau migration/counter machinery, event no-op steps,
# final fence + ckpt). Same seed (42), different GPUs (correctness only;
# timing claims come from the 30K same-GPU A/B).
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
R=$D/vos_runs
mkdir -p "$R" "$D"/logs
cd "$D"

setsid nohup $PY vos_train_driver.py --scene room --iterations 5000 --gpu 1 \
  --vos-start 9999 --final-eval subset \
  --outdir "$R"/c0_room_5k < /dev/null > "$D"/logs/vos_c0_room_5k.log 2>&1 &
echo "C0_5K pid=$!"

setsid nohup $PY vos_train_driver.py --scene room --iterations 5000 --gpu 2 \
  --vos-start 1000 --final-eval subset \
  --outdir "$R"/vos_room_5k < /dev/null > "$D"/logs/vos_vos_room_5k.log 2>&1 &
echo "VOS_5K pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
