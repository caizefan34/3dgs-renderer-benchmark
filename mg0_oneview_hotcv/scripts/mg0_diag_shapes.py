#!/usr/bin/env python3
"""Diagnostic: print param/grad shapes of the restored model at room@15000."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch

import mg0_core as C

ck = C.load_checkpoint("room", 15000)
from gaussian_model import GaussianModel
m = GaussianModel(max_sh_degree=3)
m.restore({k: ck[k] for k in
           ("active_sh_degree", "xyz", "shs", "scaling", "rotation",
            "opacity", "max_radii2D", "xyz_gradient_accum", "denom",
            "spatial_lr_scale")},
          C.training_cfg_dict())
for n in ("_xyz", "_shs", "_scaling", "_rotation", "_opacity"):
    p = getattr(m, n)
    print(f"{n}: shape={tuple(p.shape)} dtype={p.dtype}")
print("ckpt tensors:")
for k in ("xyz", "shs", "scaling", "rotation", "opacity"):
    print(f"  ckpt[{k}]: {tuple(ck[k].shape)}")
