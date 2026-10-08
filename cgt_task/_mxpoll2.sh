echo "=== NOW ==="; date +%H:%M:%S
echo "G1 procs alive:"; ps -eo pid,cmd | grep g1_compress | grep -v grep | wc -l
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_bicycle_bicycle.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_*.json
