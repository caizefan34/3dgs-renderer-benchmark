#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json, numpy as np
d=json.load(open('/home/liaoyuanjun/cgt_realscene/logs/smoke.json'))
recs=[r for r in d['records'] if r['rank']==4]
tjn=np.array([r['teacher_jac_norm'] for r in recs])
ja=np.array([r['jac_abs'] for r in recs])
jr=np.array([r['jac_rel'] for r in recs])
ng=np.array([r['n_gauss'] for r in recs])
print('teacher_jac_norm: min %.4f med %.4f max %.4f'%(tjn.min(),np.median(tjn),tjn.max()))
print('jac_abs:          min %.4f med %.4f max %.4f'%(ja.min(),np.median(ja),ja.max()))
print('jac_rel:          min %.4f med %.4f max %.4f'%(jr.min(),np.median(jr),jr.max()))
# correlation: is jac_rel huge when teacher_jac_norm tiny?
for r in sorted(recs,key=lambda x:-x['teacher_jac_norm'])[:5]:
    print('  n_gauss',r['n_gauss'],'tjn %.4f'%r['teacher_jac_norm'],'jac_abs %.4f'%r['jac_abs'],'jac_rel %.2f'%r['jac_rel'])
print('--- lowest teacher_jac_norm (near-constant tiles) ---')
for r in sorted(recs,key=lambda x:x['teacher_jac_norm'])[:5]:
    print('  n_gauss',r['n_gauss'],'tjn %.4f'%r['teacher_jac_norm'],'jac_abs %.4f'%r['jac_abs'],'jac_rel %.2f'%r['jac_rel'])
PYEOF
