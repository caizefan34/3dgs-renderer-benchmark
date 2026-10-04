#!/bin/bash
for i in $(seq 1 15); do
  line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 7)
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 7 | grep -c .)
  echo "t$i mem_util=[$line] procs=$procs"
  sleep 1
done
echo "--- probe etime ---"
ps -o etime,stat,args -p 653016 2>/dev/null | cat
