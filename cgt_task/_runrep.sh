cd /home/liaoyuanjun/cgt_realscene/final
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY build_report.py 2>&1
echo "=== head of report ==="
head -20 CGT_MASTER_REPORT.md
