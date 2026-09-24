#!/bin/bash
# Run all Phase C53-Validation analysis scripts
cd ~/3dgs-renderer-benchmark
echo "=== 1. Future Work Analysis ==="
python3 scripts/phase-c53-validation/analyze_future_work.py 2>&1
echo ""
echo "=== 2. Leakage Analysis ==="
python3 scripts/phase-c53-validation/analyze_leakage.py 2>&1
echo ""
echo "=== 3. Backward Work Analysis ==="
python3 scripts/phase-c53-validation/analyze_backward_work.py 2>&1
echo ""
echo "=== 4. Intersection-Level Analysis ==="
python3 scripts/phase-c53-validation/analyze_intersection.py 2>&1
echo ""
echo "=== 5. Age-Stratified Analysis ==="
python3 scripts/phase-c53-validation/analyze_age.py 2>&1
echo ""
echo "=== 6. Temporal Horizon Analysis ==="
python3 scripts/phase-c53-validation/analyze_horizon.py 2>&1
echo ""
echo "=== 7. Final Comparison ==="
python3 scripts/phase-c53-validation/create_final_comparison.py 2>&1
echo ""
echo "=== ALL ANALYSIS COMPLETE ==="
