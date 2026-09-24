#!/usr/bin/env bash
# C32-B: Sequential launcher for all sync ablation variants on mx.
set -e
BASE_DIR="$HOME/3dgs-renderer-benchmark"
SCRIPT="$BASE_DIR/scripts/phase-c31/c32_b_sync_ablation.py"
OUT_DIR="$BASE_DIR/results/phase-c31"
mkdir -p "$OUT_DIR"

export PATH="/usr/local/cuda/bin:$PATH"

echo "=== C32-B: All variant runs ==="
echo "Started: $(date)"
echo "---"

run() {
    local variant="$1"
    local repeats="$2"
    local out="$OUT_DIR/c32_b_${variant}.json"
    echo "[$(date +%H:%M:%S)] Launching variant $variant (repeats=$repeats)..."
    CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT" --variant "$variant" --out "$out" --repeats "$repeats"
    echo "[$(date +%H:%M:%S)] Finished variant $variant"
    echo "---"
}

# A: Baseline (3 repeats)
run "A" 3

# B1-B6: One-at-a-time (3 repeats each)
run "B1" 3
run "B2" 3
run "B3" 3
run "B4" 3
run "B5" 3
run "B6" 3

# E: No-metrics (3 repeats)
run "E" 3

# F: Correctness (3 repeats)
run "F" 3

echo "=== ALL DONE ==="
echo "Finished: $(date)"
