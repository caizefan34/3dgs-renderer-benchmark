echo "=== NOW ==="; date +%H:%M:%S
echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
echo "=== G3 alive ==="; ps -eo pid,cmd|grep g3_gpu_micro|grep -v grep|wc -l
echo "=== G1 JSON mtimes ==="; ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_*.json|awk "{print \$6,\$7,\$8}"
echo "=== G3 JSON mtimes ==="; ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g3_*.json|awk "{print \$6,\$7,\$8}"
echo "=== g3 garden log tail ==="; tail -n 3 /home/liaoyuanjun/cgt_realscene/final/logs/g3r_garden_r30k.log
