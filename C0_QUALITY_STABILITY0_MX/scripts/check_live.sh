#!/usr/bin/env bash
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
echo "watcher procs:"; pgrep -af watch_and_aggregate.sh
echo "watch log lines=$(wc -l < "$Q/logs/watch_and_aggregate.log") tail:"; tail -3 "$Q/logs/watch_and_aggregate.log"
echo "=== schedulers ==="; pgrep -af quality_scheduler.py | grep -c python
echo "=== gpu6 stdout last 2 ==="; grep -E "iter|eval|done rc" "$Q/logs/scheduler_seed4344.log" | tail -2
echo "=== gpu7 stdout last 2 ==="; grep -E "iter|eval|done rc" "$Q/logs/scheduler_seed4344_gpu7.log" | tail -2
echo "=== now ==="; date -u +%H:%M:%SZ
