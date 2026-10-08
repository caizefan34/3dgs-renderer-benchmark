#!/bin/bash
# P14 O3: room (GPU 4) and bicycle (GPU 5) in parallel, then finalize.
cd /root/epic05_p14/scripts
P=/root/miniforge3/envs/gsplat/bin/python
nohup $P p14_o3.py --scene room --gpu 4 --reps 30 \
  > /root/epic05_p14/logs/o3_room.log 2>&1 &
RPID=$!
nohup $P p14_o3.py --scene bicycle --gpu 5 --reps 30 \
  > /root/epic05_p14/logs/o3_bicycle.log 2>&1 &
BPID=$!
wait $RPID; echo "room exit=$?"
wait $BPID; echo "bicycle exit=$?"
echo "=== room tail ==="; tail -8 /root/epic05_p14/logs/o3_room.log
echo "=== bicycle tail ==="; tail -8 /root/epic05_p14/logs/o3_bicycle.log
