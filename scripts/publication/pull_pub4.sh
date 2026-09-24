#!/bin/bash
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/pub_pull4 /tmp/pub_pull4.tar
mkdir -p /tmp/pub_pull4
# all completed a-arm runs (Stage B) with results.json
for d in a0_* a1_* a2_*; do
  [ -f "$d/results.json" ] || continue
  case "$d" in
    a0_room|a0_bicycle|a0_garden|a1_room|a1_bicycle|a1_garden|a2_room|a2_bicycle|a2_garden) continue;;
  esac
  mkdir -p /tmp/pub_pull4/$d
  for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json; do
    [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_pull4/$d/
  done
done
tar cf /tmp/pub_pull4.tar -C /tmp/pub_pull4 .
echo "PACKED: $(find /tmp/pub_pull4 -type f | wc -l) files, $(ls /tmp/pub_pull4 | wc -l) runs"
