#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_LOSS_KERNEL_PROFILE.json'))
r = d['per_scene']['room']
fs = r['full_step_fused']
print("total_gpu_ms_per_iter:", fs['total_gpu_ms_per_iter'])
ks = fs['kernels']
print("kernels type:", type(ks).__name__, "len:", len(ks))
k0 = ks[0] if isinstance(ks, list) else list(ks.items())[0]
print("sample entry:", k0)
EOF
