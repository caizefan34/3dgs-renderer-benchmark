#!/usr/bin/env python
import sys, torch
sys.path.insert(0,"/root/reprotopo0_src")
import rt0_common as C, rt0_statemachine as SM
snap = torch.load("/root/reprotopo0_s0/cap/room/event_10100.pt", map_location="cpu", weights_only=False)
dev="cuda:0"
o=dict(snap)
for k in ("xyz","shs","scaling","rotation","opacity","max_radii2D","xyz_gradient_accum","denom","gaussian_ids"):
    o[k]=o[k].to(dev)
o["optimizer"]={g:{m:(d[m].to(dev) if d.get(m) is not None else None) for m in ("exp_avg","exp_avg_sq")} for g,d in snap["optimizer"].items()}
SM.reprotopo_event(o, device=dev); torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        SM.reprotopo_event(o, device=dev)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
print("=====CPU=====")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
