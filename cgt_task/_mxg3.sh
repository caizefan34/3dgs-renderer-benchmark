echo "=== g3 bicycle tail ==="; tail -n 3 /home/liaoyuanjun/cgt_realscene/final/logs/g3r_bicycle_r30k.log
echo "=== g3 room_r30k tail ==="; tail -n 3 /home/liaoyuanjun/cgt_realscene/final/logs/g3r_room_r30k.log
echo "=== g3 JSON mtimes ==="; ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g3_*.json 2>/dev/null
