#!/bin/bash
# Progress check for the 30K campaign.
cd /root/vos_traces
for s in room bicycle garden; do
  echo "=== $s ==="
  tail -2 "logs/scene_${s}_30k.log" 2>/dev/null
  for arm in c0 vos; do
    f="vos_runs/${arm}_${s}_30k/training_curve.csv"
    if [ -f "$f" ]; then
      echo "  $arm: last_eval $(tail -1 "$f")"
    else
      echo "  $arm: (no curve yet)"
    fi
  done
done
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -8
