#!/usr/bin/env python3
"""Diagnose tensor shapes / API contract for this environment."""
import os, sys, glob, json, traceback

CKPT = "results/epic05/phase7/phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt"
if not os.path.exists(CKPT):
    hits = sorted(glob.glob("results/**/iter_5000.pt", recursive=True))
    CKPT = hits[0] if hits else None
print(f"checkpoint: {CKPT}")

import torch
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("\n=== checkpoint top-level keys ===")
print(sorted(ck.keys()) if isinstance(ck, dict) else type(ck))

# find model state or flat tensors
if isinstance(ck, dict):
    ms = ck.get("model_state", {})
    if not ms and all(isinstance(v, torch.Tensor) for v in ck.values()):
        ms = ck
    print("\n=== model_state keys ===")
    print(sorted(ms.keys())[:40])
    for k in ["xyz", "scaling", "rotation", "f_dc", "opacity", "shs"]:
        if k in ms and hasattr(ms[k], "shape"):
            print(f"  {k}: {tuple(ms[k].shape)}")

# Inspect gsplat API contract: list wrapper exports
print("\n=== gsplat wrapper exports ===")
try:
    from gsplat.cuda import _wrapper as W
    names = [n for n in dir(W) if not n.startswith("_")]
    print("exports:", names[:40])
    print("count:", len(names))
except Exception:
    traceback.print_exc()
