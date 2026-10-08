echo "=== does g3 use tile_teacher? ==="
grep -n "tile_teacher\|teacher" /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g3_gpu_microbench.py | head
echo "=== garden G3 old JSON: which teacher/provenance ==="
python3 -c "import json;d=json.load(open('/home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g3_garden_r30k.json'));print('keys',list(d.keys())[:12]);print('meta',d.get('meta',{}));print('scene',d.get('scene'))" 2>/dev/null
echo "=== G1 procs alive NOW ==="
ps -eo pid,cmd | grep g1_compress | grep -v grep | wc -l
