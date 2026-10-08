cd /home/liaoyuanjun/cgt_realscene
cp src/g1_compressibility.py src/analyze_g1.py src/assemble_deliverables.py src/fail_localize.py final/SCRIPTS/ 2>/dev/null
cp run_g1.sh launch_all.sh assemble_g1.sh final/SCRIPTS/ 2>/dev/null
echo "=== SCRIPTS ==="; ls -la final/SCRIPTS/
cd final
find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
sha256sum -c MANIFEST.sha256 --quiet && echo ALL_OK
echo "=== total files ==="; find . -type f | wc -l
echo "=== du ==="; du -sh .