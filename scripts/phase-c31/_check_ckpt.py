#!/usr/bin/env python3
"""Check checkpoint format for C38."""
import torch, os

ckpt = torch.load(
    "results/epic05/phase7/phase7_room_30k_16/phase7_room_30k_16_latest.pt",
    map_location="cpu",
    weights_only=True,
)
print("Keys:", list(ckpt.keys()))
if "model_state" in ckpt:
    ms = ckpt["model_state"]
    print("model_state keys:", list(ms.keys()) if isinstance(ms, dict) else type(ms))
    if isinstance(ms, dict):
        print("xyz shape:", ms.get("xyz", ms.get("means", "?")).shape)
    print("spatial_lr_scale:", ckpt.get("spatial_lr_scale", "N/A"))
