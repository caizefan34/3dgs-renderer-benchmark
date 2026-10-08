cd /home/liaoyuanjun/cgt_realscene
# copy main analysis + missing scripts into final
cp logs/g1/G1_REAL_SCENE_COMPRESSIBILITY.json final/
cp logs/g1/g1_smoke_20tiles.json final/G1_RAW_RECORDS/ 2>/dev/null
cp run_g1.sh launch_all.sh assemble_g1.sh assemble_deliverables.py final/SCRIPTS/ 2>/dev/null
cp src/analyze_g1.py src/g1_compressibility.py final/SCRIPTS/ 2>/dev/null
# also drop a standalone copy of the analysis next to the report for visibility
cp logs/g1/G1_REAL_SCENE_COMPRESSIBILITY.json final/G1_REAL_SCENE_COMPRESSIBILITY.json
echo "=== final tree ==="
find final -maxdepth 2 -type f | sort
echo "=== file count ==="
find final -type f | wc -l