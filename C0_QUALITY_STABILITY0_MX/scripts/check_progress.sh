#!/usr/bin/env bash
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
echo "ALIVE=$(pgrep -cf quality_scheduler.py)"
echo "TRAINERS=$(pgrep -cf mx_h8_trainer.py)"
echo "=== gpu6 ==="
grep -E 'launching|done rc|complete|never cleared' "$Q/logs/scheduler_seed4344.log" | tail -8
echo "=== gpu7 ==="
grep -E 'launching|done rc|complete|never cleared' "$Q/logs/scheduler_seed4344_gpu7.log" | tail -8
echo "=== GPU util ==="
nvidia-smi --id=6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== index files ==="
ls -la "$Q"/RUN_INDEX_SEED4344*.json 2>/dev/null
