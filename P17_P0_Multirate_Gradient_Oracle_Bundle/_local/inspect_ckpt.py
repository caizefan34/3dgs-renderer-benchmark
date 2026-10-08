#!/usr/bin/env python3
"""Inspect a frozen 16K C0 checkpoint: keys, optimizer state, shapes, LRs."""
import json
import sys

import torch

p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False)
out = {"path": p, "keys": sorted(ck.keys())}
for k in ("iteration", "active_sh_degree", "spatial_lr_scale"):
    v = ck.get(k)
    try:
        v = int(v)
    except Exception:
        v = None
    out[k] = v
for name in ("xyz", "shs", "opacity", "scaling", "rotation"):
    t = ck.get(name)
    if t is not None:
        out[f"shape_{name}"] = list(t.shape)
opt = ck.get("optimizer_state_dict")
if opt is not None:
    o = {"keys": sorted(opt.keys()), "param_groups": []}
    for g in opt.get("param_groups", []):
        gg = {k: v for k, v in g.items() if k != "params"}
        o["param_groups"].append(gg)
    st = opt.get("state", {})
    shapes = {}
    for pid, d in list(st.items())[:30]:
        shapes[pid] = {k: (list(v.shape) if hasattr(v, "shape") else v)
                       for k, v in d.items() if k in ("exp_avg", "exp_avg_sq", "step")}
    o["state_entries"] = len(st)
    o["state_sample"] = shapes
    out["optimizer"] = o
print(json.dumps(out, indent=2, default=str))
