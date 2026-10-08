#!/usr/bin/env python3
"""Probe why rasterize_num_contributing_gaussians alphas != forward render_alphas."""
import os
import sys

import torch

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
C.bootstrap_gmain()
C.import_recipe()
from gsplat import rasterization

scene = "room"
dataset = C.load_dataset(scene)
model, _, meta = C.load_model_for_dataset(
    scene, dataset,
    ckpt_path=os.path.join(C.FULL30K, "gmain03_s42_room", "ckpt_30000.pt"))
cam, gt = dataset.get_item(0)

ns = C.import_recipe()
with torch.no_grad():
    r, ra, m = rasterization(
        means=model.get_xyz, quats=model.get_rotation,
        scales=model.get_scaling, opacities=model.get_opacity,
        colors=model.get_features,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.active_sh_degree,
        radius_clip=0.0, eps2d=C.EPS2D, render_mode="RGB", absgrad=False)
    nc, alphas_util = ns["rasterize_num_contributing_gaussians"](
        m["means2d"], m["conics"], m["opacities"], m["isect_offsets"],
        m["flatten_ids"], cam.image_width, cam.image_height, 16)

a_fwd = ra[0, :, :, 0].float()
a_util = alphas_util[0].float() if alphas_util.dim() == 4 else alphas_util[0]
print("shapes:", a_fwd.shape, a_util.shape, "dtypes:",
      ra.dtype, alphas_util.dtype)
d = (a_fwd - a_util).abs()
print(f"max_abs_diff={float(d.max()):.3e} mean_abs={float(d.mean()):.3e}")
print(f"n_mismatch>1e-6: {int((d > 1e-6).sum()):,} / {d.numel():,}")
print(f"n_nan fwd={int(torch.isnan(a_fwd).sum())} util={int(torch.isnan(a_util).sum())}")
# where do they differ?
if int((d > 1e-6).sum()) > 0:
    idx = (d > 1e-6).nonzero()[:5]
    for ij in idx:
        i, j = int(ij[0]), int(ij[1])
        print(f"  pix({i},{j}): fwd={float(a_fwd[i,j]):.6f} util={float(a_util[i,j]):.6f} "
              f"nc={int(nc[0,i,j])}")
# second forward with SAME inputs: is the forward itself deterministic?
r2, ra2, m2 = rasterization(
    means=model.get_xyz, quats=model.get_rotation,
    scales=model.get_scaling, opacities=model.get_opacity,
    colors=model.get_features,
    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
    width=cam.image_width, height=cam.image_height,
    tile_size=16, packed=False, sh_degree=model.active_sh_degree,
    radius_clip=0.0, eps2d=C.EPS2D, render_mode="RGB", absgrad=False)
with torch.no_grad():
    d2 = (ra2[0, :, :, 0].float() - a_fwd).abs()
    print(f"forward-vs-forward repeat: max_abs_diff={float(d2.max()):.3e}")

# also compare utility against ITS OWN forward inputs replayed (fresh call)
with torch.no_grad():
    nc2, alphas_util2 = ns["rasterize_num_contributing_gaussians"](
        m2["means2d"], m2["conics"], m2["opacities"], m2["isect_offsets"],
        m2["flatten_ids"], cam.image_width, cam.image_height, 16)
    a_util2 = alphas_util2[0].float() if alphas_util2.dim() == 4 else alphas_util2[0]
    print(f"util-vs-util repeat: max_abs_diff={float((a_util2-a_util).abs().max()):.3e}")
    print(f"util(call2) vs fwd(call2): max_abs_diff="
          f"{float((ra2[0,:,:,0].float()-a_util2).abs().max()):.3e}")
