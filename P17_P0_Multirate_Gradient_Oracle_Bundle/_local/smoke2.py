#!/usr/bin/env python3
import importlib.util
import os
import sys
import time

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
print("K0:\n", K0)
print("cam.image_width/height:", cam.image_width, cam.image_height)
print("gt shape:", gt.shape, "viewmat:", cam.viewmatrix.shape)
for d in (1, 2, 4, 8):
    Hd, Wd = H0 // d, W0 // d
    Kd = p17.scale_K(K0, d)
    print(f"\n=== d={d} H={Hd} W={Wd} ===")
    t0 = time.time()
    gt_c = (torch.as_tensor(p17.area_avg_2d(gt.cpu().numpy(), d),
                            device="cuda", dtype=torch.float32) if d > 1
            else None)
    gp, gf, frame, loss, meta = p17.render_backward_grads(
        ctx, cam, torch.as_tensor(Kd, device="cuda", dtype=torch.float32),
        Wd, Hd, p17.EPS2D_FULL / (d * d), gt_coarse=gt_c)
    print(f"d={d}: OK loss={loss:.6f} frame={tuple(frame.shape)} "
          f"gp_means={tuple(gp['means'].shape)} gf_shs={gf['shs'].numel()} "
          f"({time.time()-t0:.1f}s)")
