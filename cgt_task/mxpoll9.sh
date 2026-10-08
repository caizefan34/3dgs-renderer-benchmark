cd /home/liaoyuanjun/cgt_realscene
echo "=== room r5k ==="; tail -4 logs/g1/g1_room_r5k.log 2>&1
echo "=== room r15k ==="; tail -4 logs/g1/g1_room_r15k.log 2>&1
echo "=== garden ==="; tail -2 logs/g1/g1_garden_garden.log 2>&1
echo "=== json done ==="; ls logs/g1/*.json 2>&1
echo "=== procs ==="; pgrep -af g1_compressibility | wc -l