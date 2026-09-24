#!/bin/bash
# Run ONE window of R3 on the A100 using the fixed runner
# Usage: bash run_one_window.sh <start_iter> <output_dir>
# This script is designed to be piped via: Get-Content -Raw | ssh mx "bash -s"
set -euo pipefail

START_ITER="${1:-5000}"
DEVICE="${2:-1}"
REPO_DIR="/home/liaoyuanjun/3dgs-renderer-benchmark"
OUTDIR="${REPO_DIR}/results/reference_v1/r3/${START_ITER}"

cd "$REPO_DIR"

# Clean environment for JIT
rm -rf /tmp/torch_extensions_r3 /tmp/cuda_cache_r3 2>/dev/null || true
mkdir -p /tmp/torch_extensions_r3 /tmp/cuda_cache_r3

export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_r3"
export CUDA_CACHE_PATH="/tmp/cuda_cache_r3"
export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTDIR"

echo "=== Window ${START_ITER} starting ==="
python3 -u experiments/r3/r3_certificate_runner.py \
    --checkpoint "${REPO_DIR}/results/reference_v1/room_30k/checkpoints/iter_${START_ITER}.pt" \
    --start-iter "$START_ITER" \
    --n-iters 30 \
    --camera-sequence "${REPO_DIR}/data/camera_sequence.npy" \
    --output "$OUTDIR" \
    --pinned-commit e494458 \
    2>&1 | tee "${OUTDIR}/run.log"
