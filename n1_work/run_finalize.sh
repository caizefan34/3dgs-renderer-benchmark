#!/bin/bash
cd /root/epic05_n1
/root/miniforge3/envs/gsplat/bin/python n1_finalize.py || exit 1
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open("artifacts/N1_VERDICT.json"))
print("VERDICT:", d["verdict"])
print("blocker:", d["checkpoint_blocker"])
print(json.dumps(d["key_numbers"], indent=1))
EOF
