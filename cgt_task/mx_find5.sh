#!/bin/bash
R=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
echo "=== multiphase checkpoints ==="; ls -la $R/B1_room_multiphase/checkpoints/ 2>/dev/null
echo "=== multiphase_v2 ==="; ls -la $R/B1_room_multiphase_v2/checkpoints/ 2>/dev/null; find $R/B1_room_multiphase_v2 -maxdepth 3 -name '*.pt' 2>/dev/null
echo "=== stage_v2 ==="; find $R/B1_room_stage_v2 -maxdepth 3 2>/dev/null | head -20
echo "=== all B1 scene checkpoint .pt ==="; ls -la $R/B1_garden_30k/checkpoints/ $R/B1_bicycle_30k/checkpoints/ 2>/dev/null
