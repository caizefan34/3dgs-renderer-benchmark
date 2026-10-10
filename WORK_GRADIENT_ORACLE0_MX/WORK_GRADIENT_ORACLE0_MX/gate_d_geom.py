#!/usr/bin/env python3
"""Gate D completeness: proper-depth synthetic scene so geometry grads flow.
Camera at origin looking +z; gaussians spread in front (z in [4,9])."""
import os, sys, math, json
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

import torch
torch.manual_seed(0)
dev = "cuda:0"
C, N = 1, 4000
H, W = 128, 128
tile = 16
th, tw = math.ceil(H/tile), math.ceil(W/tile)
sh_degree = 0
K = 1

f = 120.0
Ks = torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1]]], device=dev).repeat(C,1,1)
viewmats = torch.eye(4, device=dev).unsqueeze(0).repeat(C,1,1)  # identity: cam at origin, +z forward

def make_state(grad_on):
    g = torch.Generator(device=dev).manual_seed(1)
    means = torch.zeros(N,3, device=dev)
    means[:,0] = (torch.rand(N, generator=g, device=dev)-0.5)*(W/f)*6.0
    means[:,1] = (torch.rand(N, generator=g, device=dev)-0.5)*(H/f)*6.0
    means[:,2] = 4.0 + torch.rand(N, generator=g, device=dev)*5.0
    means.requires_grad_(grad_on)
    quats = torch.zeros(N,4, device=dev); quats[:,0]=1.0; quats.requires_grad_(grad_on)
    scales = torch.full((N,3), -2.5, device=dev); scales.requires_grad_(grad_on)
    opac = torch.full((N,), 2.0, device=dev); opac.requires_grad_(grad_on)
    colors = torch.rand(N,K,3, device=dev); colors.requires_grad_(grad_on)
    return means,quats,scales,opac,colors

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R

def run(tag, tile_mask):
    means,quats,scales,opac,colors = make_state(True)
    out = R(means,quats,scales,opac,colors, sh_degree=sh_degree,
            viewmats=viewmats, Ks=Ks, width=W, height=H, tile_size=tile,
            tile_mask=tile_mask, enable_culling=False, use_higs_culling=False)
    frame = out["frame"]
    torch.cuda.synchronize()
    loss = frame.float().abs().mean()
    loss.backward()
    torch.cuda.synchronize()
    gnames=["means","quats","scales","opac","colors"]
    params=[means,quats,scales,opac,colors]
    gnz={n:int((p.grad.abs()>1e-12).sum().item()) if p.grad is not None else -1 for n,p in zip(gnames,params)}
    gfinite=all((p.grad is None) or torch.isfinite(p.grad).all().item() for p in params)
    print(f"[{tag}] loss={loss.item():.5f} frame_finite={torch.isfinite(frame).all().item()} "
          f"n_isects={out.get('metadata',{}).get('n_isects')} grad_finite={gfinite}")
    print("       grad_nonzero(>1e-12)/N: " + ", ".join(f"{n}={gnz[n]}/{N}" if gnz[n]>=0 else f"{n}=None" for n in gnames))
    return {"loss":loss.item(),"n_isects":out.get('metadata',{}).get('n_isects'),
            "grad_nonzero":gnz,"grad_finite":gfinite}

res_full = run("full", None)
mask = torch.zeros((C,th,tw), dtype=torch.bool, device=dev)
mask.reshape(C,-1)[:, ::4] = True
res_sparse = run("sparse_mask", mask)
print("\nAll geometry groups receive nonzero gradient:", all(v>0 for v in res_full["grad_nonzero"].values()))
