cd /home/liaoyuanjun/cgt_realscene/final
# regenerate manifest over all files except the manifest itself
find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 2>/dev/null
echo "=== MANIFEST line count ==="; wc -l MANIFEST.sha256
echo "=== verify ==="; sha256sum -c MANIFEST.sha256 2>&1 | tail -5
echo "=== verify pass/fail counts ==="; sha256sum -c MANIFEST.sha256 2>&1 | grep -c OK; sha256sum -c MANIFEST.sha256 2>&1 | grep -c FAILED
echo "=== top-level deliverables ==="; ls -la *.json *.md *.csv MANIFEST.sha256 2>/dev/null | awk "{print \$5,\$9}"
