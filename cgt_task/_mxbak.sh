echo "=== buggy_teacher backup ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/buggy_teacher/ 2>/dev/null || echo "NO BACKUP DIR"
echo "=== all g1_*.json in G1_RAW_RECORDS (top level) ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_*.json | awk "{print \$6,\$7,\$8}"
echo "=== ckpt of new r15k + old r15k ==="
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY -c "import json;d=json.load(open('/home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_room_r15k.json'));print('new r15k scene',d['scene'],'ckpt',d['ckpt'],'nrec',len(d['records']))"
$PY -c "import json;d=json.load(open('/home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r15k.json'));print('old r15k scene',d['scene'],'ckpt',d['ckpt'],'nrec',len(d['records']))"
