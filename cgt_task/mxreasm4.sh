cd /home/liaoyuanjun/cgt_realscene
rm -rf final; mkdir -p final
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python src/assemble_deliverables.py --logs logs/g1 --src src --out final 2>&1
echo "=== copy raw + scripts + failure-localization ==="
cp logs/g1/G1_REAL_SCENE_COMPRESSIBILITY.json final/
cp logs/g1/g1_*.json final/G1_RAW_RECORDS/ 2>/dev/null
for f in g1_bicycle_bicycle g1_garden_garden g1_room_r5k g1_room_r10k g1_room_r15k g1_room_r30k; do cp logs/g1/$f.log final/G1_RAW_RECORDS/ 2>/dev/null; done
cp logs/g1/g1_smoke_20tiles.json final/G1_RAW_RECORDS/ 2>/dev/null
cp src/assemble_deliverables.py src/fail_localize.py src/assemble_g1.sh final/SCRIPTS/ 2>/dev/null
cp run_g1.sh launch_all.sh final/SCRIPTS/ 2>/dev/null
cp final/G2_FAILURE_LOCALIZATION.json final/ 2>/dev/null
echo "=== final top-level ==="; ls final/
echo "=== count ==="; find final -type f | wc -l