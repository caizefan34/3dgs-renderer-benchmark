#!/bin/bash
cd ~/3dgs-renderer-benchmark
python3 scripts/phase-c53-discovery/analyze_predictability.py > logs/phase-c53-discovery/analyze_pred.log 2>&1
echo "predictability done"
python3 scripts/phase-c53-discovery/analyze_densification.py > logs/phase-c53-discovery/analyze_dens.log 2>&1
echo "densification done"
python3 scripts/phase-c53-discovery/analyze_temporal.py > logs/phase-c53-discovery/analyze_temp.log 2>&1
echo "temporal done"
python3 scripts/phase-c53-discovery/analyze_age.py > logs/phase-c53-discovery/analyze_age.log 2>&1
echo "age done"
python3 scripts/phase-c53-discovery/create_final_comparison.py > logs/phase-c53-discovery/analyze_final.log 2>&1
echo "final comparison done"
echo "ALL_DONE" > logs/phase-c53-discovery/analysis_complete.flag
