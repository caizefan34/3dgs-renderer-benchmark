echo "=== garden G3 valid JSON? ==="
python3 -c "import json;d=json.load(open('/home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g3_garden_r30k.json'));print('scene',d['scene'],'n_tiles',d['n_tiles'],'M_med',d['M_median']);print('per_rank',{k:round(v,3) for k,v in d['per_rank'].items()})"
echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
