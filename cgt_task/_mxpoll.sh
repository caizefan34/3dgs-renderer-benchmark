echo "=== NOW ==="; date +%H:%M:%S
echo "=== G1 procs alive ==="
ps -eo pid,etime,cmd | grep g1_compress | grep -v grep | wc -l
echo "=== G1 JSON mtimes ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_*.json
echo "=== G3 procs alive ==="
ps -eo pid,etime,cmd | grep g3_gpu | grep -v grep | awk "{print \$1,\$2}"
echo "=== g3 room_r5k log tail ==="
tail -n 4 /home/liaoyuanjun/cgt_realscene/final/logs/g3r_room_r5k.log
