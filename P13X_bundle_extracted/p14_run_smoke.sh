#!/bin/bash
# P14 O1 smoke test: 1 cell, d1 + f_q20, H=32, inline O2, isolated dirs.
cd /root/epic05_p14/scripts
P=/root/miniforge3/envs/gsplat/bin/python
P14_SMOKE=1 $P p14_o1.py --scene room --stage A --gpu 4 \
  > /root/epic05_p14/logs/smoke_o1_room.log 2>&1
echo "exit=$?"
tail -15 /root/epic05_p14/logs/smoke_o1_room.log
echo "--- smoke units ---"
ls -la /root/epic05_p14/P14_O1_UNITS_SMOKE/ /root/epic05_p14/P14_O2_UNITS_SMOKE/ 2>/dev/null
