#!/bin/bash
for f in O1 O2 O3 O4 O5; do
  echo "=== $f ==="
  tail -6 /root/vos_traces/logs/oracle_$f.log 2>/dev/null
done
echo "=== procs ==="
pgrep -f vos_replay_oracle | wc -l
