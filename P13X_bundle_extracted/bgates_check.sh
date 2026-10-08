#!/bin/bash
cd /root/epic05_p13x/runs/bicycle_diag/p13x/raw
cat > /tmp/bgates.py <<'EOF'
import json, statistics as st
x1=json.load(open('P13X_RAW_X1.json')); x2=json.load(open('P13X_RAW_X2.json'))
x3=json.load(open('P13X_RAW_X3.json')); x5=json.load(open('P13X_RAW_X5.json'))
l1=[r for r in x1 if r['lag']==250]; l2=[r for r in x2 if r['lag']==250]
l3=[r for r in x3 if r['lag']==250]
print('X1 median jaccard lag250:', st.median(r['jaccard'] for r in l1))
print('X2 median membership_jaccard:', st.median(r['membership_jaccard'] for r in l2))
print('X2 median inversion:', st.median(r['inversion_rate_median_over_tiles'] for r in l2))
print('X2 members/tile median:', st.median(r['members_per_tile_median_b'] for r in l2), 'max:', max(r['members_per_tile_max_b'] for r in l2))
for k in ('eps_0','eps_1e-06','eps_1e-05','eps_0.0001'):
    print(f"X3 {k}: dirty_isect={st.median(r[k]['dirty_intersection_fraction'] for r in l3):.4f} "
          f"dirty_tile={st.median(r[k]['dirty_tile_fraction'] for r in l3):.4f} "
          f"replay={st.median(r[k]['replay_rel_l2_raw'] for r in l3):.3e}")
print('X5 dct256 check:', max(r.get('dct256_reconstruction_rel_l2',1) for r in x5))
for K in ('4','8','16','32','64','128','256'):
    m=[r['K'][K]['means'] for r in x5]; o=[r['K'][K]['opacity'] for r in x5]
    one=[r['shadow_one_step'][f'K{K}']['delta_rel'] for r in x5]
    print(f"X5 K={K}: means cos={st.median(v['cosine'] for v in m):.6f} rel_l2={st.median(v['rel_l2'] for v in m):.5f} | "
          f"opac cos={st.median(v['cosine'] for v in o):.6f} | onestep={st.median(one):.2e}")
snap=json.load(open('P13X_RAW_SNAPSHOTS.json'))
print('n_visible range:', min(s['n_visible'] for s in snap), max(s['n_visible'] for s in snap))
print('n_isects range:', min(s['n_isects'] for s in snap), max(s['n_isects'] for s in snap))
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/bgates.py
