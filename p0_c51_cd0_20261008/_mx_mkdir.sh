#!/bin/bash
set +e
echo "=== create C51J output dir (mkdir only, no data written) ==="
mkdir -p /mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX/audit
mkdir -p /mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX/raw_logs
ls -ld /mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX /mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX/audit
echo "=== free bytes / inodes again ==="
df -B1 /mnt/storage_pool | tail -1
df -i /mnt/storage_pool | tail -1
echo "DONE"
