#!/bin/bash
# FINAL pull: P6 16 runs + b0 expansion remainder + final figtables + aggregates + p6_results.
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/pub_final /tmp/pub_final.tar
mkdir -p /tmp/pub_final/runs /tmp/pub_final/figtables /tmp/pub_final/aggregates
# P6 runs (16)
for s in drjohnson train bicycle room; do
  for a in b1as43 b1as44 c0s43 c0s44; do
    d="${a}_${s}"
    mkdir -p /tmp/pub_final/runs/$d
    for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json; do
      [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_final/runs/$d/
    done
  done
done
# b0 expansion remainder (not yet pulled: counter re-run, treehill, truck)
for d in b0_counter b0_treehill b0_truck; do
  mkdir -p /tmp/pub_final/runs/$d
  for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json; do
    [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_final/runs/$d/
  done
done
cp p6_results.json garden_seed3_table.json r13_drjohnson_review.json /tmp/pub_final/aggregates/ 2>/dev/null || true
cp /mnt/storage_pool/liaoyuanjun/pubphase/aggregates/*.json /tmp/pub_final/aggregates/
cp /mnt/storage_pool/liaoyuanjun/pubphase/figtables/* /tmp/pub_final/figtables/
tar cf /tmp/pub_final.tar -C /tmp/pub_final .
echo "PACKED: $(find /tmp/pub_final -type f | wc -l) files"
echo "runs: $(ls /tmp/pub_final/runs | wc -l), aggregates: $(ls /tmp/pub_final/aggregates | wc -l), figtables: $(ls /tmp/pub_final/figtables | wc -l)"
