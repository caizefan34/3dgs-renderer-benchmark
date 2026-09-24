#!/bin/bash
# Pull b1 expansion (10 scenes) + garden seed pairs (4 runs) + b0 gate (3, already local
# but re-pulled for consistency). Idempotent.
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/pub_pull6 /tmp/pub_pull6.tar
mkdir -p /tmp/pub_pull6
RUNS="b1_bonsai b1_counter b1_drjohnson b1_flowers b1_kitchen b1_playroom b1_stump b1_train b1_treehill b1_truck b1as43_garden b1s43_garden b1as44_garden b1s44_garden b0_room b0_bicycle b0_garden"
for d in $RUNS; do
  if [ -f "$d/results.json" ]; then
    mkdir -p /tmp/pub_pull6/$d
    for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json; do
      [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_pull6/$d/
    done
  else
    echo "SKIP (not finished): $d"
  fi
done
tar cf /tmp/pub_pull6.tar -C /tmp/pub_pull6 .
echo "PACKED: $(find /tmp/pub_pull6 -type f | wc -l) files, $(ls /tmp/pub_pull6 | wc -l) runs"
