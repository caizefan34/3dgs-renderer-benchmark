#!/bin/bash
# Collect headline metrics from the Phase-8 diagnostic + functional runs.
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/vos_traces
$PY - <<'EOF'
import json, os, glob

def peek(d):
    rj = f"vos_runs/{d}/results.json"
    if not os.path.exists(rj):
        return None
    r = json.load(open(rj))
    fe = r.get("final_eval") or {}
    tj = f"vos_runs/{d}/timing.json"
    ts = json.load(open(tj)) if os.path.exists(tj) else {}
    ph = (ts.get("phase_summary") or {}).get("optimizer", {})
    out = {"dir": d, "psnr": fe.get("psnr"), "ssim": fe.get("ssim"),
           "lpips": fe.get("lpips"), "wall_min": r.get("total_time_min"),
           "opt_mean_ms": ph.get("mean_ms")}
    sj = f"vos_runs/{d}/vos_stats.json"
    if os.path.exists(sj):
        s = json.load(open(sj))
        out["vos_steps"] = s.get("n_vos_steps")
        out["fences"] = len(s.get("fences", []))
        out["mode"] = s.get("vos_mode")
        out["region"] = s.get("timing_region")
    return out

for d in ["c0_room_5k", "vos_room_5k", "as_room_599", "vos_st_room_599",
          "ctl_st_room_599", "vos1_st_room_599", "as_room_599_twin",
          "c0_room_599", "vos_room_599"]:
    r = peek(d)
    if r:
        print(json.dumps(r))
EOF
