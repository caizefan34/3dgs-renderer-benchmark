#!/bin/bash
# Incremental pull: runs finished since pub_pull6 (b1 treehill/truck, b0 expansion wave).
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/pub_pull7 /tmp/pub_pull7.tar
mkdir -p /tmp/pub_pull7
RUNS="b1_treehill b1_truck b0_bonsai b0_counter b0_drjohnson b0_flowers b0_kitchen b0_playroom b0_stump b0_train b0_treehill b0_truck"
for d in $RUNS; do
  if [ -f "$d/results.json" ]; then
    mkdir -p /tmp/pub_pull7/$d
    for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json; do
      [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_pull7/$d/
    done
  else
    echo "SKIP (not finished): $d"
  fi
done
tar cf /tmp/pub_pull7.tar -C /tmp/pub_pull7 .
echo "PACKED: $(find /tmp/pub_pull7 -type f | wc -l) files, $(ls /tmp/pub_pull7 | wc -l) runs"
