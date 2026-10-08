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

vis1 = ctx.visible_ids(cam, K0, W0, H0, p17.EPS2D_FULL)
rad1, _, _, conics1 = ctx.producer_full(cam, K0, W0, H0, p17.EPS2D_FULL)
smin, _ = p17.sigma_min_from_conics(conics1)
d_full = p17.assign_d(smin.cpu(), 2.0)

# full-res gradient for the same population
_, g1, fr1, loss1, meta1 = p17.render_backward_grads(
    ctx, cam, K0t, W0, H0, p17.EPS2D_FULL)

# coarse d=8 gradient
Kd = torch.as_tensor(p17.scale_K(K0, 8), device="cuda", dtype=torch.float32)
_, g8, fr8, loss8, meta8 = p17.render_backward_grads(
    ctx, cam, Kd, W0 // 8, H0 // 8, p17.EPS2D_FULL / 64.0,
    gt_coarse=torch.as_tensor(
        p17.area_avg_2d(gt.cpu().numpy(), 8), device="cuda",
        dtype=torch.float32))

N = ctx.N
# population: d8-assigned (per prereg s9: all assigned gaussians)
sel8 = (d_full == 8)
in_vis1 = torch.zeros(N, dtype=torch.bool)
in_vis1[torch.as_tensor(vis1, dtype=torch.int64)] = True
print(f"pop d8(all)={int(sel8.sum())} d8&vis1={int((sel8 & in_vis1).sum())}")
print(f"  loss1={loss1:.6f} loss8={loss8:.6f}")
gp1 = {k: v.reshape(N, -1) for k, v in g1.items()}
gp8 = {k: v.reshape(N, -1) for k, v in g8.items()}
print("--- population: all d8-assigned (prereg s10 literal) ---")
for k in ("means", "scales", "quats", "opacity", "shs"):
    a = gp1[k][sel8].reshape(-1)
    b = gp8[k][sel8].reshape(-1)
    c = p17.cosine(a, b)
    nz = int((b.abs() > 0).sum().item())
    print(f"  {k:8s} cosine={'NA' if c is None else f'{c:.5f}'} "
          f"nonzero_d8={nz} norm_full={float(torch.linalg.norm(a)):.3e} "
          f"norm_d8={float(torch.linalg.norm(b)):.3e}")
print("--- population: d8-assigned AND visible at full res ---")
for k in ("means", "scales", "quats", "opacity", "shs"):
    a = gp1[k][sel8 & in_vis1].reshape(-1)
    b = gp8[k][sel8 & in_vis1].reshape(-1)
    c = p17.cosine(a, b)
    nz = int((b.abs() > 0).sum().item())
    print(f"  {k:8s} cosine={'NA' if c is None else f'{c:.5f}'} "
          f"nonzero_d8={nz} norm_full={float(torch.linalg.norm(a)):.3e} "
          f"norm_d8={float(torch.linalg.norm(b)):.3e}")
# distribution of the d8-assigned sigma values
sv = smin[pop8]
print(f"sigma_min(d8&vis1): med={float(torch.median(sv)):.2f} "
      f"p90={float(torch.quantile(sv, 0.9)):.2f} max={float(sv.max()):.2f}")
print("DONE", f"({time.time()-t0:.1f}s)")
