#!/bin/bash
cd /root/epic05_p13x/preflight/B_diagon_300/p13x/raw
cat > /tmp/sanity2.py <<'EOF'
import json
x5=json.load(open('P13X_RAW_X5.json'))
r=x5[0]
print('dct256_reconstruction_rel_l2:', r['dct256_reconstruction_rel_l2'])
for K in ('4','32','256'):
    m=r['K'][K]['means']; o=r['K'][K]['opacity']
    one=r['shadow_one_step'][f'K{K}']['delta_rel']
    print(f"K={K}: means cos={m['cosine']:.6f} rel_l2={m['rel_l2']:.6f} | "
          f"opac cos={o['cosine']:.6f} rel_l2={o['rel_l2']:.6f} | onestep={one:.2e}")
ex=r['shadow_one_step']['exact']
print('exact onestep loss_next:', ex['loss_next'], 'delta_rel:', ex['delta_rel'])
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/sanity2.py
