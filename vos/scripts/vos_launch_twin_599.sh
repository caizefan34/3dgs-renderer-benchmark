#!/bin/bash
# TWIN TEST: the SAME A_s arm run twice, sequentially on the SAME GPU.
# If the two twins are not bitwise identical, cross-run trajectory comparison
# is impossible in principle (backward atomics are order-nondeterministic),
# and Phase-8 exactness must rest on the in-process Phase-5 oracle + the
# Phase-11 quality gates measured within the twin nondeterminism envelope.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
cd "$D"
setsid nohup $PY vos_train_driver.py --scene room --iterations 599 --gpu 7 \
  --as-mode --final-eval subset \
  --outdir "$D"/vos_runs/as_room_599_twin < /dev/null > "$D"/logs/vos_as_room_599_twin.log 2>&1 &
echo "TWIN pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
