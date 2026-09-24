#!/bin/bash
set -e
cd /mnt/storage_pool/liaoyuanjun/pub_runs
rm -rf /tmp/pub_pull3 /tmp/pub_pull3.tar
mkdir -p /tmp/pub_pull3
for d in b0_room b0_bicycle b0_garden b1as43_garden b1s43_garden \
         a0_bonsai a1_bonsai a2_bonsai a0_counter a1_counter a2_counter; do
  mkdir -p /tmp/pub_pull3/$d
  for f in results.json training_results.json training_curve.csv final_status.json timing.json quality.json; do
    [ -f "$d/$f" ] && cp "$d/$f" /tmp/pub_pull3/$d/
  done
done
cp b0_gate_verdict.json b1_gate_verdict_v2.json /tmp/pub_pull3/ 2>/dev/null || true
cp /mnt/storage_pool/liaoyuanjun/pubphase/aggregates/runs_master.json /tmp/pub_pull3/ 2>/dev/null || true
cp /mnt/storage_pool/liaoyuanjun/pubphase/aggregates/p5_external.json /tmp/pub_pull3/ 2>/dev/null || true
tar cf /tmp/pub_pull3.tar -C /tmp/pub_pull3 .
echo "PACKED: $(find /tmp/pub_pull3 -type f | wc -l) files"
