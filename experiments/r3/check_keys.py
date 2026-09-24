#!/usr/bin/env python3
import torch
import os

base = "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints"
for name in ["iter_5000.pt", "iter_15000.pt"]:
    path = os.path.join(base, name)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys())
    print("%s: keys=%s" % (name, keys[:15]))
    if "model_state" in ckpt:
        print("  model_state keys: %s" % list(ckpt["model_state"].keys())[:10])
    has_active_sh_degree = "active_sh_degree" in ckpt
    has_spatial_lr_scale = "spatial_lr_scale" in ckpt
    print("  has_active_sh_degree=%s has_spatial_lr_scale=%s num_points=%s" % 
          (has_active_sh_degree, has_spatial_lr_scale, ckpt.get("num_points", "?")))
