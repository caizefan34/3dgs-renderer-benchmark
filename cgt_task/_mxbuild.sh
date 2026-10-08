echo "=== search MX for builders of benefit/failure/g4/report ==="
grep -rl "G2_BENEFIT_BUDGET\|G2_FAILURE_LOCALIZATION\|G4_BATCHING_CEILING\|h_max_to_clear\|optimistic_upper_bound" /home/liaoyuanjun/cgt_realscene/final/ 2>/dev/null | grep -vE "\.json$|MANIFEST" | head
echo "=== local cgt_task builders ==="
ls *.py *.sh 2>/dev/null | grep -iE "benefit|failure|g4|report|budget|localize|aggregate|build|reasm|manifest"
echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r5k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r15k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r30k.json | awk "{print \$6,\$7,\$8}"
