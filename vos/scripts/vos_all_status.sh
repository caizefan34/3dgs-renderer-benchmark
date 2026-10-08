#!/bin/bash
D=/root/vos_traces
echo "=== analysis ==="
tail -3 "$D"/logs/analyze.log
ls -la "$D"/analysis/
echo "=== oracle r2 ==="
for f in O1b O6 O7 O9; do
  echo "--- $f ---"
  tail -8 "$D"/logs/oracle_$f.log 2>/dev/null | grep -E "obs t=|bitwise|WROTE|Error|error|Traceback" | tail -5
done
echo "=== T5/T15 ==="
tail -1 "$D"/logs/room_T5.log 2>/dev/null
tail -1 "$D"/logs/room_T15.log 2>/dev/null
echo "=== mb_real ==="
tail -8 "$D"/logs/mb_real.log 2>/dev/null
echo "=== procs ==="
pgrep -f vos_replay_oracle | wc -l
pgrep -f vos_microbench | wc -l
pgrep -f vos_trace_driver | wc -l
pgrep -f vos_analyze_trace | wc -l
