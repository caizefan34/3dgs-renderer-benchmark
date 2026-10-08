#!/bin/bash
# Traced room run to 15100 with ckpt: gives an in-process trace+ckpt pair where
# segment 15001..15100 (post-opacity-reset stretch, no topology events in
# (15000, 15100]) has N identical to ckpt_15100 (densify_until=15000, so no
# event runs after 15000). Then the post-reset oracle test pairs cleanly.
# The densify-region (t<15000) ckpt pairing is structurally impossible: the
# trainer saves ckpts only post-event, so any same-process ckpt N differs from
# the pre-event segment N. Fence mechanics are exercised in all other runs.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
cd "$D"
setsid nohup $PY vos_trace_driver.py --scene room --iterations 15100 --gpu 3 \
  --final-eval subset --outdir "$D"/room_T15100 \
  < /dev/null > "$D"/logs/room_T15100.log 2>&1 &
echo "T15100 pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
