#!/bin/bash
cd /root/epic05_p13x/runs/room_diag/p13x/raw
cat > /tmp/gates.py <<'EOF'
import json, statistics as st
x1=json.load(open('P13X_RAW_X1.json')); x2=json.load(open('P13X_RAW_X2.json'))
x3=json.load(open('P13X_RAW_X3.json')); x5=json.load(open('P13X_RAW_X5.json'))
lag250=[r for r in x1 if r['lag']==250]
print('=== X1 (lag 250) ===')
print('median jaccard:', st.median(r['jaccard'] for r in lag250))
print('  min:', min(r['jaccard'] for r in lag250), 'max:', max(r['jaccard'] for r in lag250))
print('  median retained:', st.median(r['retained'] for r in lag250))
l2=[r for r in x2 if r['lag']==250]
print('=== X2 (lag 250) ===')
print('median membership_jaccard:', st.median(r['membership_jaccard'] for r in l2))
print('  median inversion:', st.median(r['inversion_rate_median_over_tiles'] for r in l2))
print('  frac tiles mem-hash unchanged:', st.median(r['frac_tiles_membership_hash_unchanged'] for r in l2))
print('  frac tiles order-hash unchanged:', st.median(r['frac_tiles_order_hash_unchanged'] for r in l2))
print('  members/tile median:', st.median(r['members_per_tile_median_b'] for r in l2), 'max:', max(r['members_per_tile_max_b'] for r in l2))
l3=[r for r in x3 if r['lag']==250]
print('=== X3 (lag 250) ===')
for e in ('eps_0','eps_1e-06','eps_1e-05','eps_0.0001'):
    k='eps_1e-05' if e=='eps_1e-05' else ('eps_0.0001' if '0.0001' in e else e)
    try:
        print(f"  {k}: dirty_isect={st.median(r[k]['dirty_intersection_fraction'] for r in l3):.4f} "
              f"dirty_tile={st.median(r[k]['dirty_tile_fraction'] for r in l3):.4f} "
              f"replay={st.median(r[k]['replay_rel_l2_raw'] for r in l3):.3e}")
    except KeyError:
        print(f"  {k}: KEY MISSING; keys are {list(l3[0].keys())}")
print('=== X5 ===')
print('dct256 check:', max(r.get('dct256_reconstruction_rel_l2', 1) for r in x5))
for K in ('4','8','16','32','64'):
    m=[r['K'][K]['means'] for r in x5]; o=[r['K'][K]['opacity'] for r in x5]
    one=[r['shadow_one_step'][f'K{K}']['delta_rel'] for r in x5]
    print(f"  K={K}: means cos={st.median(v['cosine'] for v in m):.6f} rel_l2={st.median(v['rel_l2'] for v in m):.5f} | "
          f"opac cos={st.median(v['cosine'] for v in o):.6f} rel_l2={st.median(v['rel_l2'] for v in o):.5f} | "
          f"onestep={st.median(one):.2e}")
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/gates.py
