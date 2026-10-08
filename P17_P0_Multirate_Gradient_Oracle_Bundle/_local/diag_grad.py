import importlib.util
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
spec = importlib.util.spec_from_file_location(
    "p17", "/root/epic05_p17/p17_p0_oracle.py")
p17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p17)

import numpy as np
import torch

t0 = time.time()
ctx = p17.SceneCtx("room")
ci = 12
ctx.cur_ci = ci
cam, gt = ctx.cam_gt(ci)
H0, W0 = int(cam.image_height), int(cam.image_width)
K0 = cam.K.detach().cpu().numpy().astype(np.float64)
K0t = torch.as_tensor(K0, device="cuda", dtype=torch.float32)
Kd = torch.as_tensor(p17.scale_K(K0, 8), device="cuda", dtype=torch.float32)

_, g8, fr8, loss8, meta8 = p17.render_backward_grads(
    ctx, cam, Kd, W0 // 8, H0 // 8, p17.EPS2D_FULL / 64.0,
    gt_coarse=torch.as_tensor(
        p17.area_avg_2d(gt.cpu().numpy(), 8), device="cuda",
        dtype=torch.float32))
vis8 = ctx.visible_ids(cam, p17.scale_K(K0, 8), W0 // 8, H0 // 8,
                       p17.EPS2D_FULL / 64.0)
N = ctx.N
print("d8 loss", loss8, "n_vis", vis8.shape[0])
for k in ("means", "scales", "quats", "opacity", "shs"):
    t = g8[k]
    nz = int((t.abs() > 0).sum().item())
    print(f"  {k:8s} nonzero={nz} max={float(t.abs().max()):.3e} "
          f"norm={float(torch.linalg.norm(t)):.3e}")
# how many of the d8-assigned gaussians are visible at d=8?
_, _, _, conics1 = ctx.producer_full(cam, K0, W0, H0, p17.EPS2D_FULL)
smin, _ = p17.sigma_min_from_conics(conics1)
d_full = p17.assign_d(smin.cpu(), 2.0)
sel8 = (d_full == 8)
vis8t = torch.as_tensor(vis8, dtype=torch.int64)
in_vis = torch.zeros(N, dtype=torch.bool)
in_vis[vis8t] = True
overlap = int((sel8 & in_vis).sum().item())
print(f"d8 assigned={int(sel8.sum())} visible_d8={int(sel8.sum() and in_vis[sel8].sum().item())}")
print("DONE", f"({time.time()-t0:.1f}s)")
