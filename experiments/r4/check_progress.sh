#!/bin/bash
echo "=== Training Progress ==="
echo ""

# Room candidate (GPU 3)
echo -n "Room candidate_c: "
grep -c "skip_mask" ~/r4_room_candidate.log 2>/dev/null
echo -n "  Last: "
grep "ITER" ~/r4_room_candidate.log 2>/dev/null | tail -1

# Bicycle candidate (GPU 5)
echo -n "Bicycle candidate_c: "
grep -c "skip_mask" ~/r4_bicycle_candidate.log 2>/dev/null
echo -n "  Last: "
grep "ITER" ~/r4_bicycle_candidate.log 2>/dev/null | tail -1

# Garden candidate (GPU 6)
echo -n "Garden candidate_c: "
grep -c "skip_mask" ~/r4_garden_candidate.log 2>/dev/null
echo -n "  Last: "
grep "ITER" ~/r4_garden_candidate.log 2>/dev/null | tail -1

# Baselines
for log in /mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/*_baseline.log; do
  scene=$(basename "$log" _baseline.log)
  echo -n "$scene baseline: "
  grep "ITER" "$log" 2>/dev/null | tail -1
done

echo ""
echo "=== Running processes ==="
ps aux | grep r4_train_wrapper | grep -v grep | wc -l

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null
