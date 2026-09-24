#!/bin/bash
export PYTHONNOUSERSITE=1
cd "$HOME/3dgs-renderer-benchmark"
PYTHON="$HOME/miniforge3/envs/anysplat/bin/python"
MERGED="/mnt/storage_pool/liaoyuanjun/r3_1_full"
echo "=== Float64 forensics on available data (5K+15K) ==="
"$PYTHON" experiments/r3/r3_float64_forensics.py --input-dir "$MERGED" --output "$MERGED/float64_forensics.json" 2>&1
echo "Forensics exit=$?"
echo "=== Copy results to audit package ==="
AUDIT="$HOME/3dgs-renderer-benchmark/audit_packages/candidate_c_r3_1_final"
cp "$MERGED/aggregated_summary.json" "$AUDIT/analysis/" 2>/dev/null && echo "copied aggregated_summary"
cp "$MERGED/float64_forensics.json" "$AUDIT/analysis/" 2>/dev/null && echo "copied float64_forensics"
cp "$MERGED/5000/final_decision.json" "$AUDIT/analysis/decision_5000.json" 2>/dev/null && echo "copied decision_5000"
cp "$MERGED/15000/final_decision.json" "$AUDIT/analysis/decision_15000.json" 2>/dev/null && echo "copied decision_15000"
echo "=== DONE ==="
