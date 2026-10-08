#!/usr/bin/env bash
d=/dev/shm/c0_quality_stability/runs
echo "=== run dir contents ==="
ls -la "$d"/room_s43_a0/ 2>/dev/null
echo "--- bicycle ---"
ls -la "$d"/bicycle_s43_c0/ 2>/dev/null
echo "=== any log files, tail ==="
for f in "$d"/room_s43_a0/*.log "$d"/bicycle_s43_c0/*.log "$d"/room_s43_a0/*.txt "$d"/bicycle_s43_c0/*.txt; do
  if [ -f "$f" ]; then echo "--- $f ---"; tail -5 "$f"; fi
done
echo "=== gpu_samples line count + last ==="
for x in room_s43_a0 bicycle_s43_c0; do
  f="$d/$x/gpu_samples.csv"; echo "$x lines=$(wc -l < "$f" 2>/dev/null)"; tail -1 "$f" 2>/dev/null
done
echo "=== now ==="; date -u +%Y-%m-%dT%H:%M:%SZ
