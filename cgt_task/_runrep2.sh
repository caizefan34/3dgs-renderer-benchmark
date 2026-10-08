cd /home/liaoyuanjun/cgt_realscene/final
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY build_report.py 2>&1
echo "=== report head ==="
head -16 CGT_MASTER_REPORT.md
echo "=== report line count ==="
wc -l CGT_MASTER_REPORT.md
