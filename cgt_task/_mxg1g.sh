echo "=== NOW ==="; date +%H:%M:%S
echo "G1 alive:"; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_room_r30k.json|awk "{print \$6,\$7,\$8}"
echo "=== r30k log tail ==="; tail -n 3 /home/liaoyuanjun/cgt_realscene/final/logs/g1_room_r30k.log
