/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python - <<'PYEOF'
import json, numpy as np
d=json.load(open('/home/liaoyuanjun/cgt_realscene/logs/smoke.json'))
recs=d['records']
TILE=16
for rank in [4,8,16]:
    rows=[r for r in recs if r['rank']==rank]
    ja=np.array([r['jac_abs'] for r in rows]); rr=np.array([r['resp_range'] for r in rows])
    thr=np.maximum(0.5*rr/TILE, 0.02/TILE)
    ok=(ja<=thr)
    print(f"rank {rank}: jac_abs med={np.median(ja):.2e} max={ja.max():.2e} | hybrid-jac gate pass {int(ok.sum())}/{len(rows)}={ok.mean()*100:.0f}% | thr med={np.median(thr):.2e}")
PYEOF