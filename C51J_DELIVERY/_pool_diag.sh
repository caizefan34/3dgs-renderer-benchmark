#!/bin/bash
set +e
echo "=== TIME ==="; date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "=== WHO fills the pool (top-level dirs by size, read-only du) ==="
sudo -n du -sh /mnt/storage_pool/* 2>/dev/null | sort -rh | head -15 || du -sh /mnt/storage_pool/liaoyuanjun/* 2>/dev/null | sort -rh | head -15
echo "=== liaoyuanjun top dirs ==="
du -sh /mnt/storage_pool/liaoyuanjun/* 2>/dev/null | sort -rh | head -12
echo "=== any big reclaimable caches under liaoyuanjun (pip/torch/hf) ==="
du -sh /mnt/storage_pool/liaoyuanjun/.cache 2>/dev/null | head
find /mnt/storage_pool/liaoyuanjun -maxdepth 3 -type d \( -name ".cache" -o -name "__pycache__" -o -name "torch_cache" \) 2>/dev/null | head -10
echo "=== is there ANY scratch/tmp area with space? ==="
df -h /tmp /dev/shm 2>/dev/null | tail -3
echo "DONE"
