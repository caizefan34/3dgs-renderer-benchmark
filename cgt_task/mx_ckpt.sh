#!/bin/bash
R=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
echo "=== checkpoint structure (room 30k) ==="
$PY - <<'PYEOF'
import torch
ck=torch.load('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/iter_30000.pt', map_location='cpu', weights_only=False)
print('type', type(ck))
if isinstance(ck, dict):
    for k,v in ck.items():
        try: print(k, tuple(v.shape) if hasattr(v,'shape') else type(v))
        except: print(k, type(v))
PYEOF
echo "=== c53_workload_statistics (room 30k) keys ==="
$PY - <<'PYEOF'
import json
d=json.load(open('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/c53_workload_statistics.json'))
def walk(o,p=''):
    if isinstance(o,dict):
        for k in list(o)[:12]:
            print(p+k, type(o[k]).__name__, (len(o[k]) if hasattr(o[k],'__len__') else ''))
            if isinstance(o[k],dict) and p.count('.')<2: walk(o[k],p+k+'.')
walk(d)
PYEOF
