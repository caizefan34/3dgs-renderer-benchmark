#!/usr/bin/env bash
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
echo "=== scheduler log sizes ==="
ls -la "$Q/logs/"
echo "=== gpu6 log tail (trainer stdout interleaved) ==="
tail -15 "$Q/logs/scheduler_seed4344.log"
echo "=== gpu7 log tail ==="
tail -15 "$Q/logs/scheduler_seed4344_gpu7.log"
echo "=== now ==="; date -u +%Y-%m-%dT%H:%M:%SZ
