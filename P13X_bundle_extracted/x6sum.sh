#!/bin/bash
cat > /tmp/x6sum.py <<'EOF'
import json
r = json.load(open('/root/epic05_p13x/artifacts/x6_room/P13X_X6_ORACLE_RAW.json'))
print('e2e_baseline_ms:', r['e2e_baseline_ms'])
print('components:', json.dumps(r['component_ms_median']))
print('bwd regression:', json.dumps(r['backward_regression']))
print('dct_ms:', json.dumps(r['dct_transform_ms']))
print('dirty:', json.dumps(r['dirty_fraction_eps1e5_lag250']))
for k, v in r['oracles'].items():
    print(f"{k}: T={v['T_ms']:.3f} speedup={v['speedup_vs_e2e']:.3f}")
print('per-camera sample:')
for row in r['per_camera'][:3]:
    print(' ', {k: (round(v,3) if isinstance(v,float) else v) for k,v in row.items()})
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/x6sum.py
