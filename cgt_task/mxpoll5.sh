cd /home/liaoyuanjun/cgt_realscene
echo "=== per-combo logs ==="
ls -la logs/g1/*.log 2>&1
echo "=== room r5k log tail ==="
tail -8 logs/g1/g1_room_r5k.log 2>&1
echo "=== garden log tail ==="
tail -8 logs/g1/g1_garden_garden.log 2>&1
echo "=== python proc cpu ==="
ps -eo pid,etimes,pcpu,comm,args | grep g1_compress | grep -v grep | head