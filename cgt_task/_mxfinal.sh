echo "=== any of my procs still running? ==="
ps -eo pid,cmd | grep -E "g1_compress|g2_cert|g3_gpu|g2_rig" | grep -v grep | wc -l
echo "=== MX final manifest verify ==="
cd /home/liaoyuanjun/cgt_realscene/final
sha256sum -c MANIFEST.sha256 2>&1 | grep -c OK
sha256sum -c MANIFEST.sha256 2>&1 | grep -c FAILED
echo "=== key deliverable mtimes (all should be ~12:0x) ==="
ls -la --time-style=+%H:%M CGT_MASTER_VERDICT.json CGT_MASTER_REPORT.md G2_RIGOROUS_CERTIFICATE.json G3_GPU_MICROBENCH.json MANIFEST.sha256 | awk "{print \$6,\$7,\$8}"
echo "=== verdict overall ==="
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY -c "import json;v=json.load(open('CGT_MASTER_VERDICT.json'));print(v['overall']);print('gpus',v['gpus_used'])"
