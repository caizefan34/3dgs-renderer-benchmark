#!/bin/bash
# Inspect timing.json phase_ms structure (per-iteration series or aggregate?)
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/vos_traces
$PY - <<'EOF'
import json
t = json.load(open("vos_runs/vos_room_5k/timing.json"))
pm = t["phase_ms"]
print("phase_ms type:", type(pm).__name__)
if isinstance(pm, dict):
    for k, v in list(pm.items())[:6]:
        print(f"  {k}: {type(v).__name__}", end="")
        if isinstance(v, list):
            print(f" len={len(v)} head={v[:3]}", end="")
        print()
elif isinstance(pm, list):
    print("  len:", len(pm), "head:", pm[:2])
print("optimizer_ms:", t["optimizer_ms"], "total_wall_s:", t["total_wall_s"])
print("mean_iter_ms:", t["mean_iter_ms"], "iters_per_s:", t["iterations_per_s"])
q = json.load(open("vos_runs/vos_room_5k/quality.json"))
print("quality keys full:", sorted(q.keys()))
EOF
