echo "=== g3 syntax check ==="
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python -c "import ast; ast.parse(open('/home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g3_gpu_microbench.py').read()); print('g3 OK')"
echo "=== G1 procs alive ==="
ps -eo pid,etime,cmd | grep g1_compress | grep -v grep | awk "{print \$1,\$2}"
echo "=== NOW ==="; date +%H:%M:%S
echo "=== G1 JSON mtimes ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_*.json | awk "{print \$6,\$7}"
