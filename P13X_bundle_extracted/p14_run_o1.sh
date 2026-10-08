#!/bin/bash
# P14 O1 Stage A: room (GPU 4) and bicycle (GPU 5) in parallel.
cd /root/epic05_p14/scripts
P=/root/miniforge3/envs/gsplat/bin/python
STAGE="$1"
TAG="$2"
nohup $P p14_o1.py --scene room --stage "$STAGE" --gpu 4 \
  > /root/epic05_p14/logs/o1${TAG}_room.log 2>&1 &
RPID=$!
nohup $P p14_o1.py --scene bicycle --stage "$STAGE" --gpu 5 \
  > /root/epic05_p14/logs/o1${TAG}_bicycle.log 2>&1 &
BPID=$!
wait $RPID; echo "room exit=$?"
wait $BPID; echo "bicycle exit=$?"
echo "=== room tail ==="; tail -6 /root/epic05_p14/logs/o1${TAG}_room.log
echo "=== bicycle tail ==="; tail -6 /root/epic05_p14/logs/o1${TAG}_bicycle.log
