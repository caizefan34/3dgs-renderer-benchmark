#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json, numpy as np
d=json.load(open('/home/liaoyuanjun/cgt_realscene/logs/smoke.json'))
recs=d['records']
print('scene',d['scene'],'n',d['n_records'],'gate',d['gate'])
for rank in [4,8,16]:
    rows=[r for r in recs if r['rank']==rank]
    psnr=np.array([r['psnr'] for r in rows]); mx=np.array([r['max_abs'] for r in rows]); jac=np.array([r['jac_rel'] for r in rows]); ng=np.array([r['n_gauss'] for r in rows])
    print(f"rank {rank}: n={len(rows)} n_gauss[min/med/max]={ng.min()}/{int(np.median(ng))}/{ng.max()} psnr[med={np.median(psnr):.1f} min={psnr.min():.1f} max={psnr.max():.1f}] max_abs[med={np.median(mx):.4f}] jac_rel[med={np.median(jac):.4f}]")
    # pass rate at gate
    ok=np.array([r['psnr']>=30 and r['max_abs']<=0.02 and r['jac_rel']<=0.05 for r in rows])
    print(f"   gate-pass {int(ok.sum())}/{len(rows)} = {ok.mean()*100:.0f}%")
PYEOF
