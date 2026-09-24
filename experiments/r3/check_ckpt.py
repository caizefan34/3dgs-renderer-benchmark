#!/usr/bin/env python3
import torch
import os

base = "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints"
for name in ["iter_5000.pt", "iter_15000.pt", "iter_30000.pt"]:
    path = os.path.join(base, name)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys())
    is_flat = "model_state" not in ckpt
    print("%s: format=%s keys=%s" % (name, "flat" if is_flat else "wrapped", str(keys[:8])))
    if is_flat:
        x = ckpt.get("xyz", None)
        print("  xyz=%s" % str(x.shape if isinstance(x, torch.Tensor) else "N/A"))
        print("  num_points=%s active_sh_degree=%s" % (ckpt.get("num_points","?"), ckpt.get("active_sh_degree","?")))
    else:
        ms = ckpt.get("model_state", {})
        x = ms.get("xyz", None)
        print("  xyz=%s" % str(x.shape if isinstance(x, torch.Tensor) else "N/A"))
        print("  num_points=%s sh_degree=%s" % (ms.get("num_points","?"), ms.get("sh_degree","?")))
