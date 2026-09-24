#!/bin/bash
# R3.1 Complete Post-Run Pipeline - single-shot execution
# Runs after all 9 sub-batches complete
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

REPO="$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
PARALLEL_DIR="/mnt/storage_pool/liaoyuanjun/r3_1_parallel"
MERGED_DIR="/mnt/storage_pool/liaoyuanjun/r3_1_full"

cd "$REPO"

echo "=== Step 0: Check all sub-batches complete ==="
complete=0
for d in 5000_A 5000_B 5000_C 15000_A 15000_B 15000_C 30000_A 30000_B 30000_C; do
    if [ -f "$PARALLEL_DIR/$d/certificate_correctness.json" ]; then
        complete=$((complete+1))
        echo "  [OK] $d"
    else
        echo "  [MISSING] $d"
    fi
done
echo "  Complete: $complete/9"
if [ "$complete" -lt 9 ]; then
    echo "Not all sub-batches complete. Aborting."
    exit 1
fi

echo ""
echo "=== Step 1: Merge sub-batch outputs ==="
"$PYTHON" experiments/r3/merge_r3_1_subbatches.py

echo ""
echo "=== Step 2: Aggregate analysis ==="
"$PYTHON" experiments/r3/r3_analyze.py --input-dir "$MERGED_DIR" --output "$MERGED_DIR"

echo ""
echo "=== Step 3: Per-window decision ==="
for w in 5000 15000 30000; do
    echo "  --- Decision for window $w ---"
    "$PYTHON" experiments/r3/r3_decision.py --input "$MERGED_DIR/$w"
done

echo ""
echo "=== Step 4: Cross-window decision ==="
"$PYTHON" experiments/r3/r3_decision.py --input "$MERGED_DIR"

echo ""
echo "=== Step 5: Float64 forensics ==="
"$PYTHON" experiments/r3/r3_float64_forensics.py --input-dir "$MERGED_DIR" --output "$MERGED_DIR/float64_forensics.json"

echo ""
echo "=== Step 6: Copy analysis to audit package ==="
AUDIT="$REPO/audit_packages/candidate_c_r3_1_final"
cp "$MERGED_DIR/aggregated_summary.json" "$AUDIT/analysis/" 2>/dev/null || true
cp "$MERGED_DIR/final_decision.json" "$AUDIT/analysis/" 2>/dev/null || true
cp "$MERGED_DIR/float64_forensics.json" "$AUDIT/analysis/" 2>/dev/null || true
for w in 5000 15000 30000; do
    cp "$MERGED_DIR/$w/final_decision.json" "$AUDIT/analysis/decision_$w.json" 2>/dev/null || true
done
cp "$REPO/reports/r3_1/vec_gate_report.json" "$AUDIT/analysis/" 2>/dev/null || true

echo ""
echo "=== Step 7: Build audit package ==="
"$PYTHON" audit_packages/build_r3_1_audit.py

echo ""
echo "=== R3.1 Post-run pipeline complete ==="
echo "Results in: $MERGED_DIR"
echo "Audit package in: $AUDIT"
