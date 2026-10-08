echo "=== all py in SCRIPTS ==="
ls /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/*.py 2>/dev/null | xargs -n1 basename
echo "=== all py in final root ==="
ls /home/liaoyuanjun/cgt_realscene/final/*.py 2>/dev/null | xargs -n1 basename
echo "=== grep builders ==="
grep -rl "G4_BATCHING_CEILING\|G2_FAILURE_LOCALIZATION\|CGT_MASTER_REPORT" /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/ /home/liaoyuanjun/cgt_realscene/final/*.py 2>/dev/null
echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r5k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r15k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r30k.json | awk "{print \$6,\$7,\$8}"
