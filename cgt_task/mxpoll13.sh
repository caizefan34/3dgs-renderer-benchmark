cd /home/liaoyuanjun/cgt_realscene
echo "=== json done ==="; ls logs/g1/*.json 2>&1
echo "=== room r10k ==="; tail -2 logs/g1/g1_room_r10k.log 2>&1
echo "=== room r30k ==="; tail -2 logs/g1/g1_room_r30k.log 2>&1
echo "=== bicycle ==="; tail -2 logs/g1/g1_bicycle_bicycle.log 2>&1
echo "=== procs ==="; pgrep -af g1_compressibility | wc -l