echo "=== g1_compressibility.py mtime (fix time) ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g1_compressibility.py
echo "=== G2 aggregate + dense + rigorous mtimes ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G2_CONTINUOUS_DOMAIN_CERTIFICATE.json /home/liaoyuanjun/cgt_realscene/final/G2_CERT/*.json
echo "=== master verdict + status + report mtimes ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/CGT_MASTER_VERDICT.json /home/liaoyuanjun/cgt_realscene/final/G2_G5_STATUS.json /home/liaoyuanjun/cgt_realscene/final/CGT_MASTER_REPORT.md /home/liaoyuanjun/cgt_realscene/final/MANIFEST.sha256
echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
