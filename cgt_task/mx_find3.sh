#!/bin/bash
R=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
echo "=== B1_room_multiphase ==="; find $R/B1_room_multiphase -maxdepth 3 -name '*.ply' 2>/dev/null | head
echo "=== B1_room_multiphase_v2 ==="; find $R/B1_room_multiphase_v2 -maxdepth 3 -name '*.ply' 2>/dev/null | head
echo "=== B1_room_stage_v2 ==="; find $R/B1_room_stage_v2 -maxdepth 3 -name '*.ply' 2>/dev/null | head
echo "=== B1_room_30k ply ==="; find $R/B1_room_30k -name '*.ply' 2>/dev/null | head
echo "=== sample ply header (room 30k) ==="
f=$(find $R/B1_room_30k -name 'point_cloud.ply' 2>/dev/null | head -1); echo "$f"; ls -la "$f" 2>/dev/null
echo "=== storage_pool top-level (find prior-art/profile jsons) ==="
find /mnt/storage_pool/liaoyuanjun -maxdepth 2 \( -iname '*PRIOR_ART*' -o -iname '*RESIDUAL_PROFILE*' -o -iname '*RESEARCH_FRONTIER*' -o -iname '*COST*' \) 2>/dev/null | head -20
