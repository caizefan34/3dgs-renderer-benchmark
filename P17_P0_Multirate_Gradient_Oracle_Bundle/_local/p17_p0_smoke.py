#!/usr/bin/env python3
"""P17 smoke: run the full per-camera pipeline for ONE probe camera only."""
import importlib.util
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
spec = importlib.util.spec_from_file_location(
    "p17", "/root/epic05_p17/p17_p0_oracle.py")
p17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p17)

import numpy as np
import torch
import time

SCENE = "room"
OUT = "/root/epic05_p17"
os.makedirs(OUT, exist_ok=True)
p17.verify_identity(OUT, SCENE)
ctx = p17.SceneCtx(SCENE)
camset = p17.build_camera_set(ctx, OUT)
ci = camset["probe_ids"][0]
ctx.cur_ci = ci
cam, gt = ctx.cam_gt(ci)
H0, W0 = int(cam.image_height), int(cam.image_width)
K0 = cam.K.detach().cpu().numpy().astype(np.float64)
K0t = torch.as_tensor(K0, device="cuda", dtype=torch.float32)
camK = {d: torch.as_tensor(p17.scale_K(K0, d), device="cuda",
                           dtype=torch.float32) for d in p17.DS}

t0 = time.time()
ref_grads, ref_frame, ref_loss, ref_meta = p17.render_backward_grads(
    ctx, cam, K0t, W0, H0, p17.EPS2D_FULL)
t1 = time.time()
orb_grads, orb_frame, orb_loss, orb_meta = p17.render_backward_grads(
    ctx, cam, K0t, W0, H0, p17.EPS2D_FULL)
t2 = time.time()
rgb_max = float((ref_frame - orb_frame).abs().max())
loss_rel = abs(ref_loss - orb_loss) / (abs(ref_loss) + 1e-30)
print(f"SMOKE d1 parity: rgb_max={rgb_max:.3e} loss_rel={loss_rel:.3e} "
      f"({t1-t0:.1f}s + {t2-t1:.1f}s)")
grad_cos = {}
for key in ("means", "scales", "quats", "opacity", "sh0", "shN"):
    grad_cos[key] = p17.cosine(ref_grads[key], orb_grads[key])
print("SMOKE cosines:", {k: (f"{v:.7f}" if v is not None else None)
                         for k, v in grad_cos.items()})
vis1 = ctx.visible_ids(cam, K0, W0, H0, p17.EPS2D_FULL)
print("SMOKE n_visible:", vis1.shape[0])

counts1, n_isects1 = ctx.intersect_counts(cam, K0, W0, H0, p17.EPS2D_FULL,
                                          vis1)
print("SMOKE n_isects_full:", n_isects1,
      "counts sum:", int(counts1.sum()),
      "top counts:", np.sort(counts1)[-5:])
_, _, _, conics1 = ctx.producer(cam, K0, W0, H0, p17.EPS2D_FULL, vis1)
smin, smax = p17.sigma_min_from_conics(conics1)
sm = smin.cpu().numpy()
print("SMOKE sigma_min (coarse-px at d=1 = full px): "
      f"min={sm.min():.4f} p10={np.percentile(sm,10):.4f} "
      f"median={np.median(sm):.4f} p90={np.percentile(sm,90):.4f} "
      f"max={sm.max():.4f}")
d_full = p17.assign_d(smin.cpu(), 2.0)
print("SMOKE tau=2 assignment:",
      {f"d{d}": int((d_full == d).sum().item()) for d in (1, 2, 4, 8)})

t3 = time.time()
Lb, La = p17.dense_adam_step(ctx, cam, K0, W0, H0)
t4 = time.time()
print(f"SMOKE dense step: L_before={Lb:.6f} L_after={La:.6f} "
      f"G={Lb-La:.3e} ({t4-t3:.1f}s)")
# audit losses sanity
t5 = time.time()
LAB = p17.audit_losses(ctx, camset["audit_ids"])
t6 = time.time()
print("SMOKE audit losses:", [f"{x:.6f}" for x in LAB], f"({t6-t5:.1f}s)")
print(f"SMOKE total so far: {time.time()-t0:.1f}s")
