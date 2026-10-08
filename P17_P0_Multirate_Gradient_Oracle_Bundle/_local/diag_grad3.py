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
N = ctx.N

vis1 = ctx.visible_ids(cam, K0, W0, H0, p17.EPS2D_FULL)
rad1, _, _, conics1 = ctx.producer_full(cam, K0, W0, H0, p17.EPS2D_FULL)
smin, _ = p17.sigma_min_from_conics(conics1)
d_full = p17.assign_d(smin.cpu(), 2.0)
sel8 = (d_full == 8)

# FULL-res gradient
ctx.restore_initial()
img1, out1 = ctx.render_train(cam, K0t, W0, H0, p17.EPS2D_FULL)
loss1 = ctx.fused_loss(img1, gt, gt_coarse=None)
print("loss1", loss1.item())
loss1.backward()
g1 = ctx.grad_groups(per_gauss=False)
ctx.model.optimizer.zero_grad(set_to_none=True)

# d=8 gradient
ctx.restore_initial()
img8, out8 = ctx.render_train(
    cam, Kd, W0 // 8, H0 // 8, p17.EPS2D_FULL / 64.0)
gt8 = torch.as_tensor(p17.area_avg_2d(gt.cpu().numpy(), 8),
                      device="cuda", dtype=torch.float32)
loss8 = ctx.fused_loss(img8, gt, gt_coarse=gt8)
print("loss8", loss8.item())
loss8.backward()
g8 = ctx.grad_groups(per_gauss=False)
ctx.model.optimizer.zero_grad(set_to_none=True)

in_vis1 = torch.zeros(N, dtype=torch.bool)
in_vis1[torch.as_tensor(vis1, dtype=torch.int64)] = True
pop8 = sel8 & in_vis1
print(f"pop d8&vis1: {int(pop8.sum())}")

print("--- d=1 vs d=8, pop d8&vis1 (full-res grad vs d8 grad) ---")
for k in ("means", "scales", "quats", "opacity", "shs"):
    per = g1[k].numel() // N
    pm = pop8.repeat_interleave(per)
    a = g1[k][pm]
    b = g8[k][pm]
    c = p17.cosine(a, b)
    print(f"  {k:8s} cosine={'NA' if c is None else f'{c:.5f}'} "
          f"norm_full={float(torch.linalg.norm(a)):.3e} "
          f"norm_d8={float(torch.linalg.norm(b)):.3e}")

print("--- d=1 self vs d=8 self (same source, both cameras) ---")
# control: d1 vs d1 (should be ~1)
a = g1["means"][pop8.repeat_interleave(3)]
print("  control d1/d1 means cosine:", p17.cosine(a, a))
print("DONE", f"({time.time()-t0:.1f}s)")
