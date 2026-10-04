#!/bin/bash
for i in $(seq 1 60); do
  p=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 7 2>/dev/null | grep .)
  if [ -n "$p" ]; then
    pid=$(echo "$p" | awk -F, '{print $1}' | tr -d ' ')
    echo "t$i CAUGHT pid=$pid mem=$p"
    ps -o pid,ppid,etime,stat,args -p "$pid" 2>/dev/null | cat
    ls -l /proc/$pid/cwd 2>/dev/null
  fi
  sleep 0.5
done
echo "DONE"
