#!/bin/bash
for log in /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/*.log; do
  scene=$(basename "$log")
  echo -n "$scene: "
  tail -1 "$log" 2>/dev/null | head -c 200
  echo
done
echo "SEP"
echo "Room candidate:"
tail -1 ~/r4_room_candidate.log 2>/dev/null | head -c 200
echo
echo "Bicycle candidate:"
tail -1 ~/r4_bicycle_candidate.log 2>/dev/null | head -c 200
echo
echo "SEP"
grep -c "skip_mask" ~/r4_room_candidate.log 2>/dev/null
