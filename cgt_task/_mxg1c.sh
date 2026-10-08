echo "G1 alive:"; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r5k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r15k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r30k.json | awk "{print \$6,\$7,\$8}"
