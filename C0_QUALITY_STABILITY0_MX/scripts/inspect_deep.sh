#!/usr/bin/env bash
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
d=/dev/shm/c0_quality_stability/runs/room_s43_a0
echo "=== final_status.json ==="; cat "$d/final_status.json"
echo ""; echo "=== memory.json ==="; cat "$d/memory.json"
echo ""; echo "=== quality.json ==="; cat "$d/quality.json"
echo ""; echo "=== training_curve.csv ==="; cat "$d/training_curve.csv"
echo ""; echo "=== results.json deep structure ==="
$PY - "$d/results.json" <<'EOF'
import json,sys
r=json.load(open(sys.argv[1]))
pi=r.get("per_iter")
print("per_iter type:", type(pi).__name__, "len:", len(pi) if pi is not None else None)
if isinstance(pi,list) and pi:
    print("per_iter[0]:", pi[0]); print("per_iter[-1]:", pi[-1])
de=r.get("densification_events")
print("densification_events type:", type(de).__name__, "len:", len(de) if de is not None else None)
if isinstance(de,list) and de:
    print("densification_events[0]:", de[0]); print("densification_events[-1]:", de[-1])
er=r.get("eval_rows")
print("eval_rows len:", len(er) if er else None)
if er: print("eval_rows[0]:", er[0])
print("binary_identity:", r.get("binary_identity"))
print("renderer:", r.get("renderer"))
EOF
