#!/bin/bash
cat > /tmp/p12ref2.py <<'EOF'
import json
r = json.load(open('/root/epic05_p12/artifacts/P12A_SYSTEM_RUNS.json'))
v = r['per_scene']['room']['C0_FUSED']
def pick(d, depth=0):
    if not isinstance(d, dict):
        return d
    return {k: pick(x, depth+1) for k, x in d.items()
            if any(t in k for t in ('iter', 'ms', 'arm', 'mean', 'wall', 'phase'))}
print(json.dumps(pick(v), indent=1)[:2500])
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/p12ref2.py
