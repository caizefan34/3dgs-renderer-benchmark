#!/bin/bash
# Watch for the FULL DSH-H timing run to complete (reps=5, samples=100).
f=/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/artifacts/timing_summary.json
log=/mnt/storage_pool/liaoyuanjun/p3h_timing2.log
for i in $(seq 1 90); do
  if [ -f "$f" ] && grep -q '"reps": 5' "$f" && grep -q '"samples_per_rep": 100' "$f"; then
    echo "FULL_DONE"
    break
  fi
  if grep -q Traceback "$log" 2>/dev/null; then
    echo "FAILED"
    break
  fi
  sleep 10
done
echo "=== live procs ==="
ps -eo etime,cmd --width=200 | grep p3_h_run
echo "=== rows (want ~31 each) ==="
wc -l /mnt/storage_pool/liaoyuanjun/higs_p3h_cache/artifacts/timing_rows_V*.csv 2>/dev/null
echo "=== summary tag/reps ==="
grep -E '"tag"|"reps"|"samples_per_rep"' "$f" 2>/dev/null