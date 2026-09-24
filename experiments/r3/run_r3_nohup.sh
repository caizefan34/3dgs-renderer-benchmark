#!/bin/bash
# R3 — nohup-based launch (survives SSH disconnects)
set -euo pipefail
DEVICE="${1:-1}"
REPO_DIR="/home/liaoyuanjun/3dgs-renderer-benchmark"
OUTPUT_BASE="${REPO_DIR}/results/reference_v1/r3"
CAM_SEQ="${REPO_DIR}/data/camera_sequence.npy"

cd "$REPO_DIR"

# CUDA/JIT isolation
rm -rf /tmp/torch_extensions_r3 /tmp/cuda_cache_r3
mkdir -p /tmp/torch_extensions_r3 /tmp/cuda_cache_r3
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_r3"
export CUDA_CACHE_PATH="/tmp/cuda_cache_r3"
export CUDA_VISIBLE_DEVICES="$DEVICE"

# Preflight
python3 experiments/r3/r3_sigma_min.py > /tmp/r3_preflight.log 2>&1
echo "Preflight done (see /tmp/r3_preflight.log)"

# Run each window sequentially
for START_ITER in 5000 10000 15000; do
    CKPT="${REPO_DIR}/results/reference_v1/room_30k/checkpoints/iter_${START_ITER}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "WARNING: $CKPT not found. Skipping."
        continue
    fi
    OUTDIR="${OUTPUT_BASE}/${START_ITER}"
    mkdir -p "$OUTDIR"
    echo ""
    echo "=== Window ${START_ITER} ==="
    nohup python3 experiments/r3/r3_certificate_runner.py \
        --checkpoint "$CKPT" \
        --start-iter "$START_ITER" \
        --n-iters 30 \
        --camera-sequence "$CAM_SEQ" \
        --output "$OUTDIR" \
        --pinned-commit e494458 \
        > "${OUTDIR}/run.log" 2>&1 &
    PID=$!
    echo "  PID=$PID  log=${OUTDIR}/run.log"
    wait $PID
    echo "  [DONE] Exit code: $?"
done

# Aggregate
echo ""
echo "=== Aggregating ==="
python3 experiments/r3/r3_analyze.py \
    --input-dir "$OUTPUT_BASE" \
    --output "$OUTPUT_BASE/" > "${OUTPUT_BASE}/aggregate.log" 2>&1 || true

# Decision
echo "=== Decision ==="
python3 experiments/r3/r3_decision.py \
    --input "$OUTPUT_BASE/" > "${OUTPUT_BASE}/decision.log" 2>&1 || true

echo ""
echo "=== R3 Complete ==="
echo "Outputs: $OUTPUT_BASE"
echo "Individual run logs: ${OUTPUT_BASE}/*/run.log"
