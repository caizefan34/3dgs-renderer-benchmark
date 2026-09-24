#!/bin/bash
# Pull Stage A run artifacts into a single tar on mx
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/stagea_pull
mkdir -p /tmp/stagea_pull
for d in a0_room a1_room a2_room a0_bicycle a1_bicycle a2_bicycle a0_garden a1_garden a2_garden; do
  mkdir -p /tmp/stagea_pull/$d
  for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json memory.json; do
    [ -f "$d/$f" ] && cp "$d/$f" /tmp/stagea_pull/$d/
  done
done
tar cf /tmp/stagea_all.tar -C /tmp/stagea_pull .
echo "PACKED: $(find /tmp/stagea_pull -type f | wc -l) files"
tar tf /tmp/stagea_all.tar | head -12
