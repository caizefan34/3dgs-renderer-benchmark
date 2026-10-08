#!/usr/bin/env bash
cd /root/epic05_dpgr0
for i in $(seq 1 120); do
  done=0; line=""
  for cell in room@10000 room@15000 bicycle@10000 bicycle@15000; do
    log="logs/phase_${cell}_gpu2_frozen.log"
    [ -f "$log" ] || log="logs/phase_${cell}_gpu3_frozen.log"
    [ -f "$log" ] || log="logs/phase_${cell}_gpu4_frozen.log"
    [ -f "$log" ] || log="logs/phase_${cell}_gpu5_frozen.log"
    c=$(grep "PHASE " "$log" 2>/dev/null | grep -v CELL_DONE | wc -l)
    nd=$(grep "PHASE " "$log" 2>/dev/null | grep -v CELL_DONE | grep -c "same_gpu_deterministic\": false")
    d=$(grep "PHASE_CELL_DONE" "$log" 2>/dev/null | wc -l)
    line="$line | $cell:s$c/nd$nd/d$d"
    if [ "$d" -ge 1 ]; then done=$((done+1)); fi
  done
  echo "[$(date -u +%H:%M:%S)] done=$done/4 $line"
  if [ "$done" -ge 4 ]; then echo "ALL_CELLS_DONE"; break; fi
  sleep 90
done