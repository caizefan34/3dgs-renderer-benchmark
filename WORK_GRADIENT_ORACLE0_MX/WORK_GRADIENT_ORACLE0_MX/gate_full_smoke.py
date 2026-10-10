#!/usr/bin/env python3
"""Full source-consistent HiGS dynamic render test.
PYTHONNOUSERSITE=1 ensures gsplat resolves entirely from source (77ab983).
Prebuilt gsplat_scene_cuda on path avoids JIT rebuild."""
import os, sys, glob, math, json

BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

import torch
torch.manual_seed(0)
dev = "cuda:0"

import gsplat
print("gsplat base:", gsplat.__file__)

# Verify isect_tiles from source has tile_mask
from gsplat.cuda._wrapper import isect_tiles
import inspect
sig = inspect.signature(isect_tiles)
print("isect_tiles has tile_mask:", "tile_mask" in sig.parameters)
print("isect_tiles module:", isect_tiles.__module__ if hasattr(isect_tiles,'__module__') else '?')
import gsplat.cuda._wrapper as w
print("_wrapper file:", w.__file__)

# Verify rendering has _maybe_evaluate_sh
from gsplat.rendering import _maybe_evaluate_sh
print("rendering._maybe_evaluate_sh: OK")

# Native scene ext
import gsplat_scene_cuda
print("gsplat_scene_cuda attrs:", [a for a in dir(gsplat_scene_cuda) if not a.startswith('_')])

C, N = 1, 3000
H, W = 128, 128
tile = 16
th, tw = math.ceil(H/tile), math.ceil(W/tile)
sh_degree = 0
K = (sh_degree+1)**2

def make_state(requires_grad):
    g = torch.Generator(device=dev).manual_seed(0)
    means = (torch.rand(N,3, generator=g, device=dev)-0.5)*3
    means[:,2]+=5.0
    quats = torch.zeros(N,4, device=dev); quats[:,0]=1.0
    scales = torch.full((N,3), -3.0, device=dev)
    opac = torch.full((N,), 2.0, device=dev)
    colors = torch.rand(N,K,3, device=dev)
    if requires_grad:
        for p in (means,quats,scales,opac,colors): p.requires_grad_(True)
    return means,quats,scales,opac,colors

viewmats = torch.eye(4, device=dev).unsqueeze(0).repeat(C,1,1)
viewmats[:,2,3] = -5.0
f = 100.0
Ks = torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1]]], device=dev).repeat(C,1,1)

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R

def run(tag, tile_mask):
    means,quats,scales,opac,colors = make_state(True)
    try:
        out = R(means,quats,scales,opac,colors, sh_degree=sh_degree,
                viewmats=viewmats, Ks=Ks, width=W, height=H, tile_size=tile,
                tile_mask=tile_mask, enable_culling=False, use_higs_culling=False)
        frame = out["frame"]
        meta = out.get("metadata",{}) or {}
        n_isects = meta.get("n_isects")
        sampled = meta.get("sampled_tile_ratio")
        torch.cuda.synchronize()
        loss = frame.float().abs().mean()
        loss.backward()
        torch.cuda.synchronize()
        gnames=["means","quats","scales","opac","colors"]
        params=[means,quats,scales,opac,colors]
        gshape={n:(tuple(p.grad.shape) if p.grad is not None else None) for n,p in zip(gnames,params)}
        gfinite=all((p.grad is None) or torch.isfinite(p.grad).all().item() for p in params)
        gnz={n:int((p.grad.abs()>0).sum().item()) if p.grad is not None else 0 for n,p in zip(gnames,params)}
        print(f"[{tag}] OK frame={tuple(frame.shape)} finite={torch.isfinite(frame).all().item()} "
              f"loss={loss.item():.5f} n_isects={n_isects} sampled_ratio={sampled}")
        print(f"       grad_finite={gfinite} grad_nonzero={gnz}")
        return {"ok":True,"n_isects":n_isects,"sampled":sampled,"loss":loss.item(),
                "gshape":gshape,"gfinite":gfinite,"gnz":gnz}
    except Exception as e:
        import traceback
        print(f"[{tag}] FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"ok":False,"error":str(e)}

res_full = run("full", None)
mask = torch.zeros((C,th,tw), dtype=torch.bool, device=dev)
mask.reshape(C,-1)[:, ::4] = True
res_sparse = run("sparse_mask", mask)

print("\n=== GATE E: sparse workload reduction ===")
if res_full.get("ok") and res_sparse.get("ok"):
    nf, ns = res_full["n_isects"], res_sparse["n_isects"]
    print(f"n_isects full={nf} sparse={ns}")
    if nf and ns:
        print(f"reduction={1-ns/nf:.2%}  (expect ~75% for 25% tiles)")

out = {"native_loaded":hasattr(gsplat_scene_cuda,"pack_gaussian_inference_scene"),
       "full":res_full,"sparse":res_sparse}
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S0R2_SMOKE_RESULT.json","w") as fp:
    json.dump(out, fp, indent=2)
print("\nSaved.")
