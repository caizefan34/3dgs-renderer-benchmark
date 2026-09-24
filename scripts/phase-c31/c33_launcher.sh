#!/usr/bin/env bash
# C33: Track A (GPU 0) + Track D (GPU 1) parallel launcher
set -e
BASE_DIR="$HOME/3dgs-renderer-benchmark"
OUT_DIR="$BASE_DIR/results/phase-c31"
mkdir -p "$OUT_DIR"
export PATH="/usr/local/cuda/bin:$PATH"

echo "=== C33: Parallel launcher ==="
echo "Started: $(date)"

# Track A — CUDA Query Overhead (GPU 0, ~3-5 min)
echo "[$(date +%H:%M:%S)] Launching Track A on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 \
    "$BASE_DIR/scripts/phase-c31/c33_a_query_overhead.py" \
    --out "$OUT_DIR/c33_a_query_overhead.json" \
    --gpu 0 &
PID_A=$!

# Track D — Workload Observation (GPU 1, ~15-20 min)
echo "[$(date +%H:%M:%S)] Launching Track D on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 \
    "$BASE_DIR/scripts/phase-c31/c33_d_workload_obs.py" \
    --out "$OUT_DIR/c33_d_workload_data.json" \
    --gpu 0 \
    --n_iters 933 &
PID_D=$!

# Wait for both
echo "Waiting for Track A (PID=$PID_A)..."
wait $PID_A
echo "[$(date +%H:%M:%S)] Track A finished (exit=$?)"

echo "Waiting for Track D (PID=$PID_D)..."
wait $PID_D
echo "[$(date +%H:%M:%S)] Track D finished (exit=$?)"

echo "=== ALL DONE ==="
echo "Finished: $(date)"
