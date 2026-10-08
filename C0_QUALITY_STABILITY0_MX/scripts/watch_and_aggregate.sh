#!/usr/bin/env bash
# Waits for both schedulers to finish (no quality_scheduler.py alive), then
# runs the aggregator and prints the verdict. Idempotent / safe to re-run.
set -u
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
echo "[watch] start $(date -u +%H:%M:%SZ)"
while true; do
  alive=$(pgrep -cf quality_scheduler.py)
  train=$(pgrep -cf mx_h8_trainer.py)
  done6=$(grep -c 'done rc' "$Q/logs/scheduler_seed4344.log")
  done7=$(grep -c 'done rc' "$Q/logs/scheduler_seed4344_gpu7.log")
  echo "[watch] $(date -u +%H:%M:%SZ) sched=$alive trainers=$train done6=$done6 done7=$done7 (target 6+6=12)"
  if [ "$alive" -eq 0 ] && [ "$train" -eq 0 ]; then
    echo "[watch] all schedulers/trainers exited; running aggregation"
    break
  fi
  sleep 120
done
echo "=== full pipeline ==="
bash "$Q/scripts/run_pipeline.sh"
echo "[watch] done $(date -u +%H:%M:%SZ)"
