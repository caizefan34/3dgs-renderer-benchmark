#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
d=json.load(open('/home/liaoyuanjun/cgt_realscene/logs/smoke.json'))
for r in d['records'][:3]:
    print(r['rank'], r.get('err'), 'n_gauss', r['n_gauss'])
PYEOF
