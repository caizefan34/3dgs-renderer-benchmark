#!/bin/bash
echo "=== fused_ssim __init__.py ==="
cat /root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim/__init__.py
echo "=== C0 run identity (c0_s42_room) ==="
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
r = json.load(open('/root/3dgs-publication-epic05/full30k/c0_s42_room/results.json'))
print(json.dumps(r['binary_identity'], indent=1))
print(json.dumps(r['renderer'], indent=1))
for arm in ('a0','a1'):
    rr = json.load(open(f'/root/3dgs-publication-epic05/full30k/{arm}_s42_room/results.json'))
    print(arm, rr['binary_identity']['so_sha256'], rr['binary_identity']['core_sha256'], rr['renderer']['env'])
gm = json.load(open('/root/3dgs-publication-epic05/full30k/gmain03_s42_room/results.json'))
print('gmain', gm['binary_identity']['so_sha256'])
import os
print('=== data scenes ===')
for s in ["bicycle","bonsai","counter","drjohnson","flowers","garden","kitchen","playroom","room","stump","train","treehill","truck"]:
    p = f'/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/{s}/sparse/0/points3D.bin'
    print(s, os.path.exists(p))
EOF
