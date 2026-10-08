cd /home/liaoyuanjun/cgt_realscene
echo "=== is G2_FAILURE_LOCALIZATION present? ==="
ls -la final/G2_FAILURE_LOCALIZATION.json 2>&1
echo "=== regenerate it (in case missing) ==="
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python src/fail_localize.py --inputs 'logs/g1/g1_*.json' --out final/G2_FAILURE_LOCALIZATION.json 2>&1 | head -4
echo "=== final top-level ==="; ls final/
echo "=== regen MANIFEST ==="
cd final
find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
sha256sum -c MANIFEST.sha256 --quiet && echo ALL_OK
wc -l MANIFEST.sha256