#!/bin/bash
set +e
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
echo "=== which GPU currently has most free, and its free MiB ==="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -2
echo "=== room dataset scale (SfM points, images) ==="
ls /mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/room/sparse/0 2>/dev/null | head
echo "=== historical baseline_room peak (from Stage5 jsons) ==="
grep -rohE "\"peak_gpu_memory_mib\":\s*[0-9.]+|\"peak_vram_mib\":\s*[0-9.]+" $REPO/results/a100/phase-c51-stage5/*.json 2>/dev/null | head
echo "=== does run_experiment accept an iters override + capture peak? (grep signature) ==="
grep -nE "def run_experiment|peak|reset_peak|max_memory_allocated|iterations|def train|iters" $REPO/scripts/phase-c51-stage5/benchmark.py | head -25
echo "DONE"
