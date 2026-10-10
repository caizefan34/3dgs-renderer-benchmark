#!/bin/bash
set +e
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
echo "=== how does benchmark.py build/load dataset? ==="
grep -nE "readScene|Dataset|source_path|benchmark_data|data_dir|mipnerf|repo_root|scene|import" $REPO/scripts/phase-c51-stage5/benchmark.py | head -40
echo "=== readScene in dataset.py ==="
grep -nE "def readScene|benchmark_data|official|datasets|source|mipnerf|return" $REPO/scripts/epic05/phase7/dataset.py | head -40
echo "=== benchmark_data/room contents ==="
ls /mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/room/ 2>/dev/null
echo "=== repo data dir? ==="
ls -la $REPO/data 2>/dev/null | head
echo DONE
