import importlib.util
import json
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
cam, gt = ctx.cam_gt(12)
H0, W0 = int(cam.image_height), int(cam.image_width)
K0 = cam.K.detach().cpu().numpy().astype(np.float64)
_, _, _, conics1 = ctx.producer_full(
    cam, K0, W0, H0, p17.EPS2D_FULL)
smin, smax = p17.sigma_min_from_conics(conics1)
print("conics shape", tuple(conics1.shape))
print("conics[0:5] =", conics1[:5].tolist())
print("sigmin p10/med/p90/max:",
      float(torch.quantile(smin.cpu(), 0.1)),
      float(torch.median(smin.cpu())),
      float(torch.quantile(smin.cpu(), 0.9)),
      float(smin.max()))
print("smax  p10/med/p90/max:",
      float(torch.quantile(smax.cpu(), 0.1)),
      float(torch.median(smax.cpu())),
      float(torch.quantile(smax.cpu(), 0.9)),
      float(smax.max()))
# manual per-Gaussian eigenvalue sanity: take gaussian 0
a, b, c = [float(x) for x in conics[0].tolist()]
tr, d, disc = (a + c) / 2, (a - c) / 2, ((a - c) / 2) ** 2 + b * b
print(f"g0: a={a:.4g} b={b:.4g} c={c:.4g} tr={tr:.4g} disc={disc:.4g}")
print(f"g0 lam_max={tr+disc:.4g} lam_min={tr-disc:.4g}")
print("DONE", f"({time.time()-t0:.1f}s)")
