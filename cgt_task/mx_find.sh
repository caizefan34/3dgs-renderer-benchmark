#!/bin/bash
echo "=== chkpnt .pth under storage_pool + tmp (maxdepth 6) ==="
find /mnt/storage_pool/liaoyuanjun* /tmp/3dgs-renderer-benchmark -maxdepth 6 -iname 'chkpnt*.pth' 2>/dev/null | head -40
echo "=== any point_cloud.ply under datasets/official ==="
find /tmp/3dgs-renderer-benchmark/data -name 'point_cloud.ply' 2>/dev/null | head
echo "=== garden ply file size + header ==="
ls -la /tmp/3dgs-renderer-benchmark/data/official/mipnerf360/garden/point_cloud.ply 2>/dev/null
echo "=== other garden locations ==="
find / -maxdepth 8 -path '*garden*point_cloud.ply' 2>/dev/null | head
echo "=== storage_pool liaoyuanjun dirs ==="
ls -d /mnt/storage_pool/liaoyuanjun* 2>/dev/null | head
echo "=== trained model dirs under repo (search point_cloud iteration subdirs) ==="
find /home/liaoyuanjun/3dgs-renderer-benchmark -maxdepth 4 -type d -name 'point_cloud' 2>/dev/null | head -20
