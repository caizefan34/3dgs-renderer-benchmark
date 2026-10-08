echo "=== G4 mtime + g4 script? ==="
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G4_BATCHING_CEILING.json | awk "{print \$6,\$7,\$8}"
ls /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g4* 2>/dev/null || echo "no g4 script"
echo "=== G4 batching_curve source: which g3 json ==="
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY -c "import json;d=json.load(open('/home/liaoyuanjun/cgt_realscene/final/G4_BATCHING_CEILING.json'));print('gpus',d.get('gpus_used') or d.get('host'));print('keys',list(d.keys()))"
echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
for f in g1_room_r5k g1_room_r15k g1_room_r30k; do ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/$f.json|awk "{print \$6,\$7,\$8}"; done
