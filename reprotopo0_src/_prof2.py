#!/usr/bin/env python
import sys, time, statistics, torch
sys.path.insert(0,"/root/reprotopo0_src")
import rt0_common as C, rt0_event as E, rt0_statemachine as SM
snap = torch.load("/root/reprotopo0_s0/cap/room/event_10100.pt", map_location="cpu", weights_only=False)
dev="cuda:0"

def to_dev(s):
    o=dict(s)
    for k in ("xyz","shs","scaling","rotation","opacity","max_radii2D","xyz_gradient_accum","denom","gaussian_ids"):
        if torch.is_tensor(o[k]): o[k]=o[k].to(dev)
    o["optimizer"]={g:{m:(d[m].to(dev) if d.get(m) is not None else None) for m in ("exp_avg","exp_avg_sq")} for g,d in s["optimizer"].items()}
    return o

# 1) current (CPU snap, transfers inside timer)
SM.reprotopo_event(snap, device=dev); torch.cuda.synchronize()
ts=[]
for _ in range(64):
    torch.cuda.synchronize(); t0=time.perf_counter()
    SM.reprotopo_event(snap, device=dev)
    torch.cuda.synchronize(); t1=time.perf_counter(); ts.append(t1-t0)
print("CURRENT(cpu-snap, transfer in timer) median %.4f min %.4f"%(statistics.median(ts),min(ts)))

# 2) fair: pre-load to device (transfer outside timer), like original harness
snapd = to_dev(snap)
SM.reprotopo_event(snapd, device=dev); torch.cuda.synchronize()
ts=[]
for _ in range(256):
    torch.cuda.synchronize(); t0=time.perf_counter()
    SM.reprotopo_event(snapd, device=dev)
    torch.cuda.synchronize(); t1=time.perf_counter(); ts.append(t1-t0)
print("FAIR(device-snap, transfer outside) median %.4f min %.4f"%(statistics.median(ts),min(ts)))

# original (restore outside timer)
ts=[]
for _ in range(64):
    m=E.restore_original_model(snap,device=dev)
    torch.cuda.synchronize(); t0=time.perf_counter()
    m.densify_and_prune(max_grad=snap["config"]["densify_grad_threshold"],min_opacity=snap["config"]["min_opacity"],extent=snap["scene_extent"],max_screen_size=snap["config"]["max_screen_size"],radii=snap["max_radii2D"].to(dev))
    torch.cuda.synchronize(); t1=time.perf_counter(); ts.append(t1-t0)
print("ORIGINAL(restore outside) median %.4f min %.4f"%(statistics.median(ts),min(ts)))
