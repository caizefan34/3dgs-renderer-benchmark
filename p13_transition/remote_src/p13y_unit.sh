#!/bin/bash
# P13-Y contamination policy wrapper: one unit = one logged run.
# usage: p13y_unit.sh <unit_label> <logfile> -- <cmd...>
label="$1"; log="$2"; shift 2; [ "$1" = "--" ] && shift
cont=/root/epic05_p13y/logs/P13Y_CONTAMINATION.log
{
  echo "=== UNIT $label START $(date -u +%FT%TZ) ==="
  echo "-- gpu inventory --"
  nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
  echo "-- compute apps before --"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
} >> "$cont"
"$@" > "$log" 2>&1
rc=$?
{
  echo "=== UNIT $label END $(date -u +%FT%TZ) rc=$rc ==="
  echo "-- compute apps after --"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
  echo
} >> "$cont"
exit $rc
