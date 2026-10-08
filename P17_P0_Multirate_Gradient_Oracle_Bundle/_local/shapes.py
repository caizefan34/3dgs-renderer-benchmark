#!/usr/bin/env python3
import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
spec = importlib.util.spec_from_file_location(
    "p17", "/root/epic05_p17/p17_p0_oracle.py")
p17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p17)

import numpy as np
import torch

SCENE = "room"
OUT = "/root/epic05_p17"
p17.verify_identity(OUT, SCENE)
ctx = p17.SceneCtx(SCENE)
ci = 12
ctx.cur_ci = ci
cam, gt = ctx.cam_gt(ci)
H0, W0 = int(cam.image_height), int(cam.image_width)
K0 = cam.K.detach().cpu().numpy().astype(np.float64)
K0t = torch.as_tensor(K0, device="cuda", dtype=torch.float32)
Kd = torch.as_tensor(p17.scale_K(K0, 2), device="cuda", dtype=torch.float32)
_, g_d, fr, loss, meta = p17.render_backward_grads(
    ctx, cam, Kd, W0 // 2, H0 // 2, p17.EPS2D_FULL / 4.0,
    gt_coarse=torch.as_tensor(p17.area_avg_2d(gt.cpu().numpy(), 2),
                              device="cuda", dtype=torch.float32))
for k, v in g_d.items():
    print(k, tuple(v.shape), "dim", v.dim())
