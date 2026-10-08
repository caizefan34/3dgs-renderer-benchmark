#!/bin/bash
# Whose contexts are these? Track growth over 60s + check my jobs' logs.
cd /root/vos_traces
echo "=== my scene logs ==="
for s in room bicycle garden; do
  echo "-- $s:"; tail -2 "logs/scene_${s}_30k.log" 2>/dev/null
  tail -1 "logs/vos_c0_${s}_30k.log" 2>/dev/null
done
echo "=== snapshot 1 ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
sleep 60
echo "=== snapshot 2 (60s later) ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
echo "=== host uptime / users ==="
uptime
who 2>/dev/null | head -5 || echo "(no who)"
