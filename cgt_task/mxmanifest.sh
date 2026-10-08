cd /home/liaoyuanjun/cgt_realscene
cp assemble_g1.sh assemble_deliverables.py final/SCRIPTS/ 2>/dev/null
cp src/assemble_deliverables.py final/SCRIPTS/ 2>/dev/null
cd final
find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
echo "=== MANIFEST verify ==="
sha256sum -c MANIFEST.sha256 --quiet && echo ALL_OK
echo "=== MANIFEST lines ==="; wc -l MANIFEST.sha256
echo "=== full final tree ==="
find . -type f | sort
echo "=== total size ==="; du -sh .