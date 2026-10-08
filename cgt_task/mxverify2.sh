cd /home/liaoyuanjun/cgt_realscene/final
echo "=== report first 12 lines ==="
head -12 CGT_MASTER_REPORT.md
echo "=== report has per-tile table? ==="
grep -c "0.99\|per-tile\|不可压缩\|上界" CGT_MASTER_REPORT.md
echo "=== verdict q_weighted r16 (should be ~0.99) ==="
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python -c "import json;d=json.load(open('CGT_MASTER_VERDICT.json'));print('overall',d['overall']);print('keys',list(d['G1_summary'].keys()));print('q16 sample',list(d['G1_summary']['q_weighted_r16'].items())[:2])"
echo "=== MANIFEST verify ==="
sha256sum -c MANIFEST.sha256 --quiet && echo ALL_OK
wc -l MANIFEST.sha256