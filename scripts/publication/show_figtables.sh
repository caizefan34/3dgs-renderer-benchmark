#!/bin/bash
set -e
cd /mnt/storage_pool/liaoyuanjun/pubphase/figtables
tar cf /tmp/figtables.tar .
echo "PACKED $(ls | wc -l) files"
echo "===== table1 ====="
cat table1_p1_baselines.md
echo "===== table2 ====="
cat table2_p2_ablation_wall.md
echo "===== table4 ====="
cat table4_p4_eps2d.md
