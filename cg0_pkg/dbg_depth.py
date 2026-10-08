#!/usr/bin/env python3
"""CG0 debug: verify depth convention + backprojection + compile render."""
import sys, os, math, time
sys.path.insert(0, "/root/epic05_n1")
sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree")
sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
from cg0_run import render_depth, build_compiled_model, eval_set

torch.manual_seed(0)
C.bootstrap_c0()
scene = "room"; it = 15000
ds = C.load_scene_dataset(scene)
n = len(ds)
split = P.build_split(n)
print("split:", {k: (len(v) if isinstance(v, list) else v) for k, v in split.items()})
ck = C.load_checkpoint(scene, it)
N_source = int(ck["xyz"].shape[0]); src_shd = int(ck["active_sh_degree"]); ss = float(ck["spatial_lr_scale"])
src_model = C.build_model(ck)
print("N_source", N_source, "shd", src_shd, "ss", ss)
print("src xyz range", ck["xyz"].min(0).values.tolist(), ck["xyz"].max(0).values.tolist())
print("cam0 K diag", ds.get_camera(0).K[0,0].item(), ds.get_camera(0).K[1,1].item(), "center", ds.get_camera(0).camera_center.tolist())
print("cam0 viewmatrix\n", ds.get_camera(0).viewmatrix)

ci = split["compile64"][0]
cam = ds.get_camera(ci)
gt = ds.get_gt_image(ci)
print("gt range", gt.min().item(), gt.max().item(), gt.dtype, gt.shape)
img, rad, m2d = C.render_view(src_model, cam, src_shd)
print("src render range", img.min().item(), img.max().item(), "psnr-ish")
mse = torch.mean((img-gt)**2); print("src-vs-gt mse", mse.item(), "psnr", float(20*torch.log10(1.0/torch.sqrt(mse))))

d, ni = render_depth(src_model, cam, src_shd)
print("depth stats: min", d.min().item(), "max", d.max().item(), "mean", d.mean().item(), "median", d.median().item(), "finite", torch.isfinite(d).sum().item(), "/", d.numel())

# backproject center pixel
H, W = d.shape
cu, cv = W//2, H//2
dc = d[cv, cu]
print("center depth", dc.item())
dev=cam.K.device; X = COM.backproject(cam, torch.tensor([float(cu)],device=dev), torch.tensor([float(cv)],device=dev), torch.tensor([float(dc)],device=dev))
print("backprojected center X", X.tolist())
# reproject
u,v,z,inf = APP._project(cam, X)
print("reproject u,v,z", u.tolist(), v.tolist(), z.tolist(), "in_frame", inf.tolist())

# compare to source xyz nearest
xyz = ck["xyz"]
dist = torch.norm(xyz - X[0].cpu(), dim=1)
k = dist.argmin()
print("nearest src xyz dist", dist[k].item(), "src xyz", xyz[k].tolist(), "src depth-equiv", dist[k].item())

