#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('/root/epic05_p11/dumps/P11_SOURCE_IDENTITY.json'))
w = d['fused_ssim']['window_constant_check']
print('window max_abs_diff:', w['max_abs_diff'])
print('window rel:', w.get('max_rel_diff'))
print('fused commit:', d['fused_ssim']['commit'][:12], 'drift:', repr(d['fused_ssim'].get('repo_status')))
print('env:', d['environment']['gpus'][0])
for s, rec in d['scenes'].items():
    print(s, 'ckpt sha:', rec['ckpt_30000_sha256'][:12],
          'seq sha:', (rec['camera_sequence_npy_sha256'] or '')[:12],
          'launcher meta:', bool(rec['campaign_launcher_meta']))
EOF
