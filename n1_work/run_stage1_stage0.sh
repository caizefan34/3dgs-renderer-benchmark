#!/bin/bash
set -e
cd /root/epic05_n1
PY=/root/miniforge3/envs/gsplat/bin/python
echo "=== syntax check ==="
for f in n1_core.py n1_stage1_probes.py n1_stage0_audit.py; do
  $PY -m py_compile $f && echo "OK $f"
done
echo "=== stage 1: probe list (CPU, before any probe gradient) ==="
$PY n1_stage1_probes.py 2>&1 | tail -12
echo "=== stage 0 audit: room (GPU0) + bicycle (GPU1) in parallel ==="
CUDA_VISIBLE_DEVICES=0 $PY n1_stage0_audit.py room > /root/epic05_n1/logs/stage0_room.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY n1_stage0_audit.py bicycle > /root/epic05_n1/logs/stage0_bicycle.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "room audit rc=$R0 bicycle audit rc=$R1"
echo "=== room audit tail ==="
tail -10 /root/epic05_n1/logs/stage0_room.log
echo "=== bicycle audit tail ==="
tail -10 /root/epic05_n1/logs/stage0_bicycle.log
