#!/bin/bash
echo "=== R5-A Progress ==="
echo ""

for seed in 1 2; do
  for scene in train truck; do
    for method in baseline candidate; do
      log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${seed}_${method}.log"
      echo -n "$scene seed=$seed $method: "
      if [ -f "$log" ]; then
        grep "ITER" "$log" 2>/dev/null | tail -1 | head -c 150
      else
        echo -n "NOT STARTED"
      fi
      echo
    done
  done
done

echo ""
echo "=== Running processes ==="
ps aux | grep r4_train | grep -v grep | wc -l
echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null
