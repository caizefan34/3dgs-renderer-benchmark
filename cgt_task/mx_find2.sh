#!/bin/bash
echo "=== all iteration_*/point_cloud.ply under sota_rebase_v2_runs ==="
find /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs -maxdepth 4 -name 'point_cloud.pth' 2>/dev/null | head
find /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs -maxdepth 4 -type d -name 'iteration_*' 2>/dev/null | head -40
echo "=== scene coverage: which scenes have 30k models ==="
ls -d /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/*/ 2>/dev/null | head -60
echo "=== prior-art / residual profile / cost-model assets in repo ==="
find /home/liaoyuanjun/3dgs-renderer-benchmark -maxdepth 3 \( -iname '*PRIOR_ART*' -o -iname '*RESIDUAL_PROFILE*' -o -iname '*RESEARCH_FRONTIER*' -o -iname '*COST_MODEL*' -o -iname '*STAGE_FRACTION*' \) 2>/dev/null | head -20
