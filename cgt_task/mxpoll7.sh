cd /home/liaoyuanjun/cgt_realscene
echo "=== room r5k log ==="
wc -l logs/g1/g1_room_r5k.log 2>&1; tail -4 logs/g1/g1_room_r5k.log 2>&1
echo "=== room r15k log ==="
wc -l logs/g1/g1_room_r15k.log 2>&1; tail -4 logs/g1/g1_room_r15k.log 2>&1
echo "=== python proc cpu/etime ==="
ps -eo pid,etimes,pcpu,args | grep g1_compressibility | grep -v grep | sed 's/--data.*--out/.../' 