#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
R=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
echo "=== _mx_colmap_to_json.py (head) ==="
sed -n '1,60p' /home/liaoyuanjun/3dgs-renderer-benchmark/_mx_colmap_to_json.py
echo "=== provenance.json (room 30k) ==="
$PY -c "import json;d=json.load(open('$R/B1_room_30k/provenance.json'));print(json.dumps(d,indent=1)[:1500])"
