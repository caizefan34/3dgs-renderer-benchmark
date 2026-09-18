#!/bin/bash
# Delete intermediate checkpoints, keep only iter_30000
echo "Before cleanup:"
df -h /mnt/storage_pool/liaoyuanjun/ 2>/dev/null | tail -1

# Delete all iter_*.pt except iter_30000.pt
find /mnt/storage_pool/liaoyuanjun/r4_13scene_v2/ -name 'iter_*.pt' ! -name 'iter_30000.pt' -exec rm -v {} \; 2>/dev/null | wc -l

echo "After cleanup:"
df -h /mnt/storage_pool/liaoyuanjun/ 2>/dev/null | tail -1

echo "Remaining checkpoints:"
find /mnt/storage_pool/liaoyuanjun/r4_13scene_v2/ -name 'iter_*.pt' -exec ls -lh {} \; 2>/dev/null
