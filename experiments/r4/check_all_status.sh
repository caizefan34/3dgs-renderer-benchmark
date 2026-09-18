#!/bin/bash
for log in ~/r4_13scene_v2_logs/*_baseline.log; do
  scene=$(basename "$log" _baseline.log)
  echo -n "$scene: "
  tail -1 "$log" 2>/dev/null | head -c 150
  echo
done
echo "SEP"
for log in ~/r4_13scene_v2_logs/*_candidate.log; do
  scene=$(basename "$log" _candidate.log)
  echo -n "$scene: "
  tail -1 "$log" 2>/dev/null | head -c 150
  echo
done
echo "SEP"
ps aux | grep r4_train | grep -v grep | wc -l
echo "SEP"
ps aux | grep r4_train | grep -v grep | awk '{print $NF}' | sort
