#!/bin/bash
D=/root/vos_traces
for s in room_T30 bicycle_T30 garden_T30 room_R15 bicycle_R15 garden_R15 room_R5 bicycle_R5 garden_R5; do
  echo "=== $s ==="
  ls -la "$D/$s/" 2>/dev/null | grep -Ev "^total|^d" | awk '{print $5, $9}' | grep -Ev "^$"
done
echo "=== sizes ==="
df -h /root | tail -1
