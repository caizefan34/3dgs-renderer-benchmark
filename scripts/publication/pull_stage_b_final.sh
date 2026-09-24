#!/bin/bash
# Final Stage B pull: all 39 a-arm runs (Stage A + Stage B), 6 files each.
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/pub_pull5 /tmp/pub_pull5.tar
mkdir -p /tmp/pub_pull5
ALL13="bicycle bonsai counter drjohnson flowers garden kitchen playroom room stump train treehill truck"
for s in $ALL13; do
  for a in a0 a1 a2; do
    d="${a}_${s}"
    [ -f "$d/results.json" ] || { echo "MISSING $d"; exit 1; }
    mkdir -p /tmp/pub_pull5/$d
    for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json; do
      [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_pull5/$d/
    done
  done
done
tar cf /tmp/pub_pull5.tar -C /tmp/pub_pull5 .
echo "PACKED: $(find /tmp/pub_pull5 -type f | wc -l) files, $(ls /tmp/pub_pull5 | wc -l) runs"
