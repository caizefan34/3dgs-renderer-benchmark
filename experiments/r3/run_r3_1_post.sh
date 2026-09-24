#!/bin/bash
# R3.1 Post-Run: Aggregate + Decision + Audit Package
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
RESULTS="/mnt/storage_pool/liaoyuanjun/r3_1_full"

cd "$REPO"

echo "=== Step 1: Check all windows have output ==="
ALL_READY=true
for w in 5000 15000 30000; do
    n=$(ls "$RESULTS/$w/certificate_correctness.json" 2>/dev/null | wc -l)
    if [ "$n" -lt 1 ]; then
        echo "  [WARN] $w: certificate_correctness.json not found"
        ALL_READY=false
    else
        echo "  [OK] $w: output ready"
    fi
done

if [ "$ALL_READY" = false ]; then
    echo "Not all windows ready. Aborting."
    exit 1
fi

echo ""
echo "=== Step 2: Aggregate analysis ==="
"$PYTHON" experiments/r3/r3_analyze.py --input-dir "$RESULTS" --output "$RESULTS"
echo "  Aggregated summary written."

echo ""
echo "=== Step 3: Per-window decision ==="
for w in 5000 15000 30000; do
    echo "  --- Decision for window $w ---"
    "$PYTHON" experiments/r3/r3_decision.py --input "$RESULTS/$w"
done

echo ""
echo "=== Step 4: Cross-window decision ==="
"$PYTHON" experiments/r3/r3_decision.py --input "$RESULTS"

echo ""
echo "=== Step 5: Build audit package ==="
"$PYTHON" audit_packages/build_r3_1_audit.py

echo ""
echo "=== R3.1 Post-run pipeline complete ==="
