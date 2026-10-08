#!/bin/bash
cat > /tmp/p12ref.py <<'EOF'
import json
r = json.load(open('/root/epic05_p12/artifacts/P12A_SYSTEM_RUNS.json'))
def walk(o, path=""):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "room" and isinstance(v, dict):
                hits.append((path + "/" + k, v))
            else:
                hits.extend(walk(v, path + "/" + k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            hits.extend(walk(v, f"{path}[{i}]"))
    return hits
hits = walk(r)
for p, v in hits[:4]:
    keys = [k for k in v.keys() if 'iter' in k or 'ms' in k or 'phase' in k or 'arm' in k or 'loss' in k]
    print(p, {k: v[k] for k in keys} if keys else list(v.keys())[:12])
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/p12ref.py
