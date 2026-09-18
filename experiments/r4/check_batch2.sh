#!/bin/bash
echo "=== Running processes ==="
ps aux | grep r4_train | grep -v grep | awk '{print $NF}' | sort
echo "SEP"
echo "=== Counter candidate result ==="
grep 'PSNR=' /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/counter_candidate.log 2>/dev/null | tail -5
echo "SEP"
echo "=== Stump candidate result ==="
grep 'PSNR=' /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/stump_candidate.log 2>/dev/null | tail -5
echo "SEP"
echo "=== Batch 2 logs ==="
ls /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/ 2>/dev/null | sort
echo "SEP"
echo "=== GPU usage ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null
