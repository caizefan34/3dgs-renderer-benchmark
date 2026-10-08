#!/bin/bash
# VOS Phase-4 status check
D=/root/vos_traces
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for s in room_T30 bicycle_T30 garden_T30; do
  echo "=== $s ==="
  tail -3 "$D/logs/$s.log" 2>/dev/null
  ls -la "$D/$s/vos_trace.bin" 2>/dev/null | awk '{print "trace_bin:", $5, "bytes"}'
done
for s in room_R15 bicycle_R15 garden_R15 room_R5 bicycle_R5 garden_R5; do
  echo "=== $s ==="
  tail -2 "$D/logs/$s.log" 2>/dev/null
done
echo "=== chains ==="
cat "$D/logs/chain_gpu6.log" 2>/dev/null
cat "$D/logs/chain_gpu7.log" 2>/dev/null
