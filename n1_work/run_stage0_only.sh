#!/bin/bash
cd /root/epic05_n1
PY=/root/miniforge3/envs/gsplat/bin/python
$PY -m py_compile n1_core.py n1_stage0_audit.py && echo "syntax OK"
CUDA_VISIBLE_DEVICES=0 $PY n1_stage0_audit.py room > /root/epic05_n1/logs/stage0_room.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY n1_stage0_audit.py bicycle > /root/epic05_n1/logs/stage0_bicycle.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "room audit rc=$R0 bicycle audit rc=$R1"
echo "=== room audit tail ==="
tail -12 /root/epic05_n1/logs/stage0_room.log
echo "=== bicycle audit tail ==="
tail -12 /root/epic05_n1/logs/stage0_bicycle.log
