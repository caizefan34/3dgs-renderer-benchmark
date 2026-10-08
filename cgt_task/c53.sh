#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
d=json.load(open('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/c53_workload_statistics.json'))
for k in ['2000','5000','10000','15000','20000','25000','30000']:
    if k in d.get('checkpoints',{}):
        c=d['checkpoints'][k]
        print(k, 'n_gauss',c['n_gauss'],'n_visible',c['n_visible'],'tiles_mean',round(c['tiles_mean'],2),'tiles_p99',c['tiles_p99'],'top1_mass',c['tiles_top1_mass'],'top5_mass',c['tiles_top5_mass'],'top10_mass',c['tiles_top10_mass'])
PYEOF
