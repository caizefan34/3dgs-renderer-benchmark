#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_LOSS_KERNEL_PROFILE.json'))
r = d['per_scene']['room']
print("top keys:", list(r.keys()))
for k in ('full_step_fused', 'full_step_sepssim', 'loss_only_fused'):
    fs = r[k]
    print(f"--- {k}: {list(fs.keys())}")
    for key in fs:
        v = fs[key]
        if isinstance(v, dict) and len(v) > 3:
            items = sorted(v.items(), key=lambda kv: -kv[1])[:6] \
                if all(isinstance(x, (int, float)) for x in v.values()) else list(v.items())[:6]
            print(f"    {key}: {items}")
EOF
echo "=== D1 ==="
tail -n 10 logs/D1.log 2>/dev/null
