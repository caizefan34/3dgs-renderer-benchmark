#!/usr/bin/env bash
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
echo "watcher procs:"; pgrep -af watch_and_aggregate.sh
echo "watch log:"; tail -5 "$Q/logs/watch_and_aggregate.log" 2>/dev/null || echo "no log yet"
