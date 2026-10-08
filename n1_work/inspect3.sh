#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open("/root/epic05_n1/artifacts/N1_CHECKPOINT_IDENTITY_room.json"))
print("top keys:", list(d.keys()))
rec = d["checkpoints"]["15000"]
for k, v in rec.items():
    print(f"  {k}: {v if not isinstance(v, dict) else list(v.keys())}")
a = json.load(open("/root/epic05_n1/artifacts/N1_AUDIT_RAW_room.json"))
print("audit keys:", list(a.keys()))
EOF
