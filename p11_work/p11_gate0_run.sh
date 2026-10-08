#!/bin/bash
cd /root/epic05_p11
CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/gsplat/bin/python p11_gate0_identity.py --gpu 0 \
  --out dumps/P11_SOURCE_IDENTITY.json 2>&1 | tail -4
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_SOURCE_IDENTITY.json'))
print('so sha:', d['gsplat_so']['sha256'][:16])
print('fs so sha:', d['fused_ssim']['so_sha256'][:16])
print('window bitwise equal:', d['fused_ssim']['window_constant_check']['bitwise_equal'])
print('scenes:', list(d['scenes'].keys()))
print('upstream:', d['upstream']['commit'][:12], d['upstream']['gsplat_version'])
print('trainer sha:', d['trainer']['sha256'][:16])
EOF
