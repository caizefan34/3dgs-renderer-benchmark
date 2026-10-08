#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
A = "/root/epic05_n1/artifacts"
for f in ("N1_ACTION_HETEROGENEITY.json", "N1_SKETCH_PREDICTIVITY.json", "N1_BUDGET_FRONTIER.json"):
    print("=" * 30, f)
    d = json.load(open(f"{A}/{f}"))
    print(json.dumps(d, indent=1)[:5200])
EOF
