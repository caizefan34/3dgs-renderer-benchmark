echo "=== NOW ==="; date +%H:%M:%S
echo "G1 alive:"; ps -eo pid,cmd|grep g1_compress|grep -v grep|awk "{print \$1}"
for f in g1_room_room_r5k g1_room_room_r30k; do ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/$f.json 2>/dev/null|awk "{print \$6,\$7,\$8}"; done
echo "=== room log view counts ==="
for f in g1_room_r5k g1_room_r30k; do echo -n "$f views_done="; grep -c "^\[room\] view" /home/liaoyuanjun/cgt_realscene/final/logs/$f.log 2>/dev/null; done
