#!/bin/bash
# Read key results from completed window 5000
REPO="$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)"
D="$REPO/results/reference_v1/r3/5000"
echo "=== files ==="
ls -la "$D" 2>/dev/null
echo "=== key JSON contents (truncated) ==="
python3 - <<'PYEOF'
import json, os, glob
base = os.path.expanduser("~") + "/3dgs-renderer-benchmark/results/reference_v1/r3/5000"
for f in sorted(os.listdir(base)):
    if f.endswith(".json"):
        p = os.path.join(base, f)
        try:
            with open(p) as fh:
                data = json.load(fh)
            print(f"\n=== {f} ===")
            if isinstance(data, dict):
                for k, v in list(data.items())[:10]:
                    s = str(v)
                    if len(s) > 250:
                        s = s[:250] + "..."
                    print(f"  {k}: {s}")
            else:
                print(f"  type={type(data).__name__} len={len(data) if hasattr(data,'__len__') else '?'}")
        except Exception as e:
            print(f"  ERROR reading: {e}")
PYEOF
