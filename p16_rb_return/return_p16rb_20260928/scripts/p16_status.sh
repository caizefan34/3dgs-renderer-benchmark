#!/bin/bash
# P16-RB compact read-only status digest for monitoring.
set -u
D=/root/epic05_p16rb
cd $D
P_PROF=$(grep -c "PROFILE_.*_DONE" logs/final_rerun2.log 2>/dev/null || true)
P_CUR=$(grep -c "CURVES_.*_DONE" logs/final_rerun3.log 2>/dev/null || true)
R2=$(grep -q FINAL_RERUN2_DONE logs/final_rerun2.log 2>/dev/null && echo DONE || echo pending)
R3=$(grep -qE "FINAL_RERUN3_DONE|FINAL_RERUN3_ABORT" logs/final_rerun3.log 2>/dev/null && echo DONE || echo pending)
FA_G=$(grep -cE "ANALYZE_FAIL|PATCH_FAIL|EQUIV_GATE_FAIL|PATCH_SYNTAX_FAIL" logs/final_rerun2.log 2>/dev/null || true)
FA_C=$(grep -cE "FATAL|_FAIL" logs/final_rerun3.log 2>/dev/null || true)
CH2=$(pgrep -f 'chain_final_rerun2[.]sh' >/dev/null && echo up || echo gone)
CH3=$(pgrep -f 'chain_final_rerun3[.]sh' >/dev/null && echo up || echo gone)
CURJS=$(ls curves/*/room.json curves/*/bicycle.json curves/*/garden.json 2>/dev/null | wc -l)
PROFJS=$(ls profiles/*/room_mid.json profiles/*/room_final.json profiles/*/bicycle_mid.json profiles/*/bicycle_final.json profiles/*/garden_mid.json profiles/*/garden_final.json 2>/dev/null | wc -l)
GPU2=$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | grep -m1 bcca92b9 || echo missing)
echo "prof_markers=${P_PROF}/12 curves_markers=${P_CUR} r2=${R2} r3=${R3} prof_fails=${FA_G} curve_fails=${FA_C} chain2=${CH2} chain3=${CH3} curve_jsons=${CURJS}/12 prof_jsons=${PROFJS}/24"
echo "gpu2: ${GPU2}"
grep -E "FULL|PROFILE_" logs/final_rerun2.log 2>/dev/null | tail -2 | sed 's/^/r2| /'
grep -E "curve|CURVES|train rc|point |marker" logs/final_rerun3.log 2>/dev/null | tail -3 | sed 's/^/r3| /'
