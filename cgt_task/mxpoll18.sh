cd /home/liaoyuanjun/cgt_realscene
echo "=== json count ==="; ls logs/g1/*.json 2>/dev/null | wc -l
echo "=== room r30k ==="; tail -2 logs/g1/g1_room_r30k.log 2>&1
echo "=== procs ==="; pgrep -af g1_compressibility | wc -l
echo "=== gpu5/6/7 idle? ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '6,8p'