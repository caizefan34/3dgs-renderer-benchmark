echo "G1 alive:"; ps -eo pid,cmd|grep g1_compress|grep -v grep|awk "{print \$1}"
for f in g1_room_room_r5k g1_room_room_r30k; do ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/$f.json 2>/dev/null|awk "{print \$6,\$7,\$8}"||echo "$f pending"; done
