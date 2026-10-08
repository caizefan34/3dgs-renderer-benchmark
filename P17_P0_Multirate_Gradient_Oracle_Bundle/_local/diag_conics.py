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
vis_rows = rad1[:, 0] > 0
print(f"valid rows: {int(vis_rows.sum())}/{len(vis_rows)}")

# conics for a few valid rows
valid_ids = torch.where(vis_rows)[0]
print(f"first 5 valid row ids: {valid_ids[:5].tolist()}")
print(f"conics rows: {conics1[valid_ids[:5]].tolist()}")

# sigma for valid rows only
smin_valid, _ = p17.sigma_min_from_conics(conics1[vis_rows])
print(f"sigma_min(valid) p10={float(torch.quantile(smin_valid, 0.1)):.4f} "
      f"med={float(torch.median(smin_valid)):.4f} "
      f"p90={float(torch.quantile(smin_valid, 0.9)):.4f} "
      f"max={float(smin_valid.max()):.4f}")

# conics for invalid rows
invalid_ids = torch.where(~vis_rows)[0]
print(f"first 5 invalid row ids: {invalid_ids[:5].tolist()}")
print(f"conics invalid rows: {conics1[invalid_ids[:5]].tolist()}")
smin_invalid, _ = p17.sigma_min_from_conics(conics1[~vis_rows])
print(f"sigma_min(invalid) p10={float(torch.quantile(smin_invalid, 0.1)):.4f} "
      f"med={float(torch.median(smin_invalid)):.4f} "
      f"p90={float(torch.quantile(smin_invalid, 0.9)):.4f} "
      f"max={float(smin_invalid.max()):.4f}")

# assignment with zeroed invalid rows
sig_min1, _ = p17.sigma_min_from_conics(conics1)
sig_min1[~vis_rows] = 0.0
d_full = p17.assign_d(sig_min1.cpu(), 2.0)
print(f"assignment (zeroed invalid): d1={int((d_full==1).sum())} "
      f"d2={int((d_full==2).sum())} d4={int((d_full==4).sum())} "
      f"d8={int((d_full==8).sum())}")

# assignment with only valid rows (invalid -> d1)
d_full2 = torch.ones(len(sig_min1), dtype=torch.int64)
for dd in (2, 4, 8):
    mask = (sig_min1 / dd >= 2.0) & vis_rows
    d_full2[mask] = dd
print(f"assignment (explicit valid-only): d1={int((d_full2==1).sum())} "
      f"d2={int((d_full2==2).sum())} d4={int((d_full2==4).sum())} "
      f"d8={int((d_full2==8).sum())}")

# check: are the d>1 gaussians from valid-only assignment visible at their d?
N = ctx.N
sel42 = (d_full2 >= 2)
in_vis1 = torch.zeros(N, dtype=torch.bool)
in_vis1[torch.as_tensor(vis1, dtype=torch.int64)] = True
print(f"d>1 gaussians: {int(sel42.sum())} of which visible at d=1: "
      f"{int((sel42 & in_vis1).sum())}")
print("DONE", f"({time.time()-t0:.1f}s)")
