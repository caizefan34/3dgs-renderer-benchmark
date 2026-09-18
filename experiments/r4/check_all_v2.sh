#!/bin/bash
echo "=== Running processes ==="
ps aux | grep r4_train | grep -v grep | awk '{print $NF}' | sort
echo "SEP"
echo "=== Log files ==="
ls /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/ 2>/dev/null
echo "SEP"
echo "=== Latest progress ==="
for log in /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/*_candidate.log; do
  scene=$(basename "$log" _candidate.log)
  echo -n "$scene candidate: "
  grep "ITER" "$log" 2>/dev/null | tail -1 | head -c 150
  echo
done
echo "SEP"
echo "=== Room ==="
grep 'PSNR=' ~/r4_room_candidate.log | tail -3
echo "SEP"
echo "=== Bicycle ==="
grep 'PSNR=' ~/r4_bicycle_candidate.log | tail -3
echo "SEP"
echo "=== Garden ==="
grep 'PSNR=' ~/r4_garden_candidate.log | tail -3
echo "SEP"
echo "=== Batch 2 ==="
ls /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/treehill* 2>/dev/null
ls /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/train* 2>/dev/null
ls /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/truck* 2>/dev/null
ls /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/drjohnson* 2>/dev/null
ls /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/playroom* 2>/dev/null
