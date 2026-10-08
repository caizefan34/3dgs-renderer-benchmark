echo "=== NOW ==="; date +%H:%M:%S
echo "G1 alive:"; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
for f in g1_room_r5k g1_room_r15k g1_room_r30k; do ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/$f.json|awk "{print \$6,\$7,\$8}"; done
echo "=== any ALL DONE in g1 logs ==="
grep -l "ALL DONE\|G1_OVERALL\|wrote" /home/liaoyuanjun/cgt_realscene/final/logs/g1_room*.log 2>/dev/null
tail -n 3 /home/liaoyuanjun/cgt_realscene/final/logs/g1_room_r30k.log 2>/dev/null
