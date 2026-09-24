#!/bin/bash
# Launch full R3 measurement on remote mx host (3 windows x 30 iters)
set -u
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate anyspatl
cd $HOME/3gds-renderer-benchmark

echo "=== Full R3 Measurement ===" 
echo "Start: $(date)"

for START_ITER in 5000 15000 29970; do
    OUTDIR="results/reference_v1/r3/${START_ITER}"
    mkdir -p "$OUTDIR"
    echo "--- Window ${START_ITER} @ $(date) ---"
    python3 -u experiments/r3/r3_certificate_runner.py \
        --checkpoint "results/reference_v1/room_30k/checkpoints/iter_${START_ITER}.pt" \
        --start-iter "$START_ITER" \
        --n-iters 30 \
        --camera-sequence data/camera_sequence.npy \
        --output "$OUTDIR" \
        --pinned-commit e494458 \
        > "$OUTDIR/runner.log" 2>&1
    echo "[DONE] ${START_ITER} exit=$?"
done

echo "=== Analyze ==="
python3 experiments/r3/r3_analyze.py \
    --input-dir results/reference_v1/r3 \
    --output results/reference_v1/r3/ \
    > results/reference_v1/r3/analyze.log 2>&1
echo "[ANALYZE_DONE] exit=$?"

echo "=== Complete @ $(date) ==="
