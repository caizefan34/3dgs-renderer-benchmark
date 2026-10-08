echo "=== NOW ==="; date +%H:%M:%S
echo "G1 alive:"; ps -eo pid,cmd|grep g1_compress|grep -v grep|awk "{print \$1}"
echo "=== all G1 JSON mtimes ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_*.json|awk "{print \$6,\$7,\$8}"
echo "=== room log tails (which views done) ==="
for f in g1_room_r5k g1_room_r15k g1_room_r30k; do echo "--- $f ---"; grep -c "^\[room\] view" /home/liaoyuanjun/cgt_realscene/final/logs/$f.log 2>/dev/null; done
