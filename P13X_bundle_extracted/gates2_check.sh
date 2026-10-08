#!/bin/bash
cd /root/epic05_p13x/runs/room_diag/p13x/raw
cat > /tmp/gates2.py <<'EOF'
import json, statistics as st
x4=json.load(open('P13X_RAW_X4.json')); x5=json.load(open('P13X_RAW_X5.json'))
print('=== X4 (16K window, %d samples) ===' % len(x4))
for K in ('4','8','16','32','64','128','256'):
    s=[r[f'K{K}'] for r in x4]
    print(f"  K={K}: energy={st.median(v['energy_retention_mean'] for v in s):.4f} "
          f"rel_l2_med={st.median(v['rel_l2_median'] for v in s):.4f} "
          f"cos_med={st.median(v['cosine_median'] for v in s):.4f} "
          f"frac_cos>=0.995={st.median(v['frac_tiles_cosine_ge']['0.995'] for v in s):.4f}")
print('=== X5 high K ===')
for K in ('128','256'):
    m=[r['K'][K]['means'] for r in x5]; o=[r['K'][K]['opacity'] for r in x5]
    sh=[r['K'][K]['sh'] for r in x5]
    one=[r['shadow_one_step'][f'K{K}']['delta_rel'] for r in x5]
    print(f"  K={K}: means cos={st.median(v['cosine'] for v in m):.6f} rel_l2={st.median(v['rel_l2'] for v in m):.5f} | "
          f"opac cos={st.median(v['cosine'] for v in o):.6f} | sh cos={st.median(v['cosine'] for v in sh):.6f} | "
          f"onestep={st.median(one):.2e}")
print('  top10 sign means K32:', st.median(r['K']['32']['means']['top10_sign_agreement'] for r in x5))
print('  probe_loss range:', min(r['probe_loss'] for r in x5), max(r['probe_loss'] for r in x5))
snap=json.load(open('P13X_RAW_SNAPSHOTS.json'))
print('snapshots n_visible range:', min(s['n_visible'] for s in snap), max(s['n_visible'] for s in snap))
print('snapshots n_isects range:', min(s['n_isects'] for s in snap), max(s['n_isects'] for s in snap))
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/gates2.py
