#!/usr/bin/env python3
"""Full recursive dump of checkpoint structure (top 3 levels)."""
import torch, os

path = "results/epic05/phase7/phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt"
ck = torch.load(path, map_location="cpu", weights_only=False)

def dump(obj, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}: dict({len(v)})")
                dump(v, prefix + "  ", depth + 1, max_depth)
            elif isinstance(v, torch.Tensor):
                print(f"{prefix}{k}: Tensor {tuple(v.shape)} {v.dtype}")
            else:
                print(f"{prefix}{k}: {type(v).__name__} = {v}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}[{len(obj)} items]")
        for i, item in enumerate(obj[:3]):
            dump(item, prefix + "  ", depth + 1, max_depth)
    else:
        print(f"{prefix}{type(obj).__name__} = {obj}")

dump(ck)
