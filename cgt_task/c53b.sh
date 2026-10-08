#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
d=json.load(open('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/c53_workload_statistics.json'))
print('top keys', list(d.keys()))
ck=d.get('checkpoints',{})
print('ckpt keys', list(ck.keys()))
for k in ['5000','15000','30000']:
    if k in ck:
        c=ck[k]
        print(k, {kk:c[kk] for kk in c if kk in ['n_gaussians','n_visible','tiles_mean','tiles_median','tiles_p99','tiles_max','tiles_top1_mass','tiles_top5_mass','tiles_top10_mass']})
PYEOF
