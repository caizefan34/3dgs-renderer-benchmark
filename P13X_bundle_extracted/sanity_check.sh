#!/bin/bash
cd /root/epic05_p13x/preflight/B_diagon_300/p13x/raw
cat > /tmp/sanity.py <<'EOF'
import json
x1=json.load(open('P13X_RAW_X1.json')); x2=json.load(open('P13X_RAW_X2.json'))
x3=json.load(open('P13X_RAW_X3.json')); x4=json.load(open('P13X_RAW_X4.json'))
x5=json.load(open('P13X_RAW_X5.json'))
print('X1 rows', len(x1), 'jaccard range', min(r['jaccard'] for r in x1), max(r['jaccard'] for r in x1))
print('X2 mem_jaccard range', min(r['membership_jaccard'] for r in x2), max(r['membership_jaccard'] for r in x2))
print('X2 inversion range', min(r['inversion_rate_median_over_tiles'] for r in x2), max(r['inversion_rate_median_over_tiles'] for r in x2))
print('X3 dirty_isect eps1e-5 range', min(r['eps_1e-05']['dirty_intersection_fraction'] for r in x3), max(r['eps_1e-05']['dirty_intersection_fraction'] for r in x3))
print('X3 replay eps0 range', min(r['eps_0']['replay_rel_l2_raw'] for r in x3), max(r['eps_0']['replay_rel_l2_raw'] for r in x3))
r=x4[0]; print('X4 sample K32:', {k:round(v,4) for k,v in r['K32'].items() if 'frac' not in k})
r=x5[0]; print('X5 K4 means:', {k:round(v,4) for k,v in r['K']['4']['means'].items()})
print('X5 K32 means:', {k:round(v,4) for k,v in r['K']['32']['means'].items()})
print('X5 K256 means:', {k:round(v,4) for k,v in r['K']['256']['means'].items()})
print('X5 K32 onestep:', r['shadow_one_step']['K32'])
print('X2 members/tile median:', x2[0]['members_per_tile_median_b'], 'max', x2[0]['members_per_tile_max_b'])
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/sanity.py
