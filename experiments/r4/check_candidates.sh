#!/bin/bash
echo "=== Candidate logs ==="
for log in /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/*_candidate.log; do
  scene=$(basename "$log" _candidate.log)
  echo -n "$scene: "
  tail -1 "$log" 2>/dev/null | head -c 150
  echo
done
echo "SEP"
echo "=== Running processes ==="
ps aux | grep r4_train | grep -v grep | awk '{print $NF}' | sort
echo "SEP"
echo "=== Room candidate 20K eval ==="
grep 'PSNR=' ~/r4_room_candidate.log 2>/dev/null | tail -5
