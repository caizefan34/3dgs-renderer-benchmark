#!/bin/bash
# P16-RB digest: one-line telemetry for the curves queue phase (cycle 5).
D=/root/epic05_p16rb
L=$D/logs/final_rerun3.log
pm=$(grep -c "PROFILE_.*_DONE" $D/logs/run_profiles.log 2>/dev/null)
pf=$(grep -c "rc=[1-9]" $D/logs/run_profiles.log 2>/dev/null)
cm=$(grep -c "CURVES_.*_DONE" $L 2>/dev/null)
cf=$(grep -c "rc=[1-9]" $L 2>/dev/null)
cj=$(ls $D/curves/*/*.json 2>/dev/null | wc -l)
pj=$(find $D/profiles -maxdepth 2 \( -name "*_mid.json" -o -name "*_final.json" \) 2>/dev/null | wc -l)
r2=$(grep -o "PROFILES_ALL_DONE.*" $D/logs/run_profiles.log 2>/dev/null | tail -1)
r3=$(grep -o "FINAL_RERUN3_DONE.*" $L 2>/dev/null | head -1)
chain3=$(pgrep -f "run_curves.sh" >/dev/null && echo up || echo gone)
gpu=$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits | sed -n '3p')
echo "prof_markers=$pm/12 curves_markers=$cm r2=${r2:-pending} r3=${r3:-pending} prof_fails=$pf curve_fails=$cf chain3=$chain3 curve_jsons=$cj/12 prof_jsons=$pj"
echo "gpu2: $gpu"
tail -3 $L | sed 's/^/r3| /'
tail -2 $D/logs/run_profiles.log | sed 's/^/r2| /'
