#!/bin/bash
# Pull b0 drjohnson/flowers (finished during the scheduler restart window) + counter attempt1.
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/pub_pull8 /tmp/pub_pull8.tar
mkdir -p /tmp/pub_pull8
for d in b0_drjohnson b0_flowers b0_counter_contaminated_attempt1; do
  if [ -f "$d/results.json" ]; then
    mkdir -p /tmp/pub_pull8/$d
    for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json; do
      [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_pull8/$d/
    done
  else
    echo "SKIP: $d"
  fi
done
tar cf /tmp/pub_pull8.tar -C /tmp/pub_pull8 .
echo "PACKED: $(find /tmp/pub_pull8 -type f | wc -l) files"
