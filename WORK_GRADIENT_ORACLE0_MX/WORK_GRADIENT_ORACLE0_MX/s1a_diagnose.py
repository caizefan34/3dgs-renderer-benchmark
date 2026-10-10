#!/usr/bin/env python3
"""Diagnose render quality: high-freq content, 10K ckpt, actual COLMAP cx/cy."""
import os, sys, json, math, struct
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
import torch, numpy as np
from PIL import Image
dev = torch.device("cuda:6"); torch.manual_seed(0)

def load_ckpt(ckpt):
    blob=torch.load(ckpt,map_location="cpu",weights_only=False)
    g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
    means=g["gaussians._means"].to(dev).float()
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales=g["gaussians._scales"].to(dev).float().exp()
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
    return means,quats,scales,opac,colors

cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H=396,264
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

# COLMAP sparse/0/cameras.bin is a stub (LFS pointer); skip cx/cy, use center.
c=cams[0]; scale=W/float(c["width"])
fx_c=fy_c=None

R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
vm=torch.tensor(vm,dtype=torch.float32,device=dev)

def render_with_K(K,w2,h2):
    out=R_fn(means,quats,scales,opac,colors,sh_degree=3,viewmats=vm.unsqueeze(0),Ks=K.unsqueeze(0),
             width=w2,height=h2,tile_size=16,near_plane=0.01,far_plane=1e10,
             background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    return out["frame"]

# GT
gt=None
for sub in ("eval_images","images"):
    for ext in (".JPG",".jpg",".png"):
        pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
        if os.path.exists(pp) and os.path.getsize(pp)>1000:
            gt=Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS); break
    if gt is not None: break
gt_t=torch.tensor(np.asarray(gt,np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)

def hfreq(img_t):
    g=img_t.mean(1)
    lap=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32,device=dev)
    from torch.nn.functional import conv2d
    hf=conv2d(g[None],lap[None,None],padding=1)[0]
    return hf.abs().mean().item()

def metrics(rend):
    rend=rend.permute(0,3,1,2).clamp(0,1)
    mse=(rend-gt_t).pow(2).mean().item(); psnr=-10*np.log10(max(mse,1e-12))
    corr=np.corrcoef(rend[0].cpu().numpy().ravel(),gt_t[0].cpu().numpy().ravel())[0,1]
    return psnr,corr

for tag,ckpt in [("5K","/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"),
                 ("10K","/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0010000.pt")]:
    means,quats,scales,opac,colors=load_ckpt(ckpt)
    # K with json (center principal point)
    K_center=torch.tensor([[c["fx"]*scale,0,(W-1)/2.0],[0,c["fy"]*scale,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
    r_center=render_with_K(K_center,W,H)
    pc,cc=metrics(r_center)
    print(f"\n[{tag}] N={means.shape[0]}")
    print(f"  K=center(json):  PSNR={pc:.2f} corr={cc:.3f} hfreq={hfreq(r_center):.4f}")
    print(f"  GT hfreq={hfreq(gt_t):.4f}")
