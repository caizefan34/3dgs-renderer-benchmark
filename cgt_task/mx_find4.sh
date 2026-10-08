#!/bin/bash
R=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
echo "=== B1_room_30k tree (3 levels) ==="; find $R/B1_room_30k -maxdepth 3 2>/dev/null | head -30
echo "=== B1_room_multiphase tree (3 levels) ==="; find $R/B1_room_multiphase -maxdepth 3 2>/dev/null | head -30
echo "=== B1_room_stage_v2 tree (3 levels) ==="; find $R/B1_room_stage_v2 -maxdepth 3 2>/dev/null | head -30
echo "=== any .pth in B1_* ==="; find $R/B1_room_30k $R/B1_garden_30k $R/B1_bicycle_30k -name '*.pth' 2>/dev/null | head
