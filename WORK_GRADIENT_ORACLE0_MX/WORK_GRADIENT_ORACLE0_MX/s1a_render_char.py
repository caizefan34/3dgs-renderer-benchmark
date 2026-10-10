#!/usr/bin/env python3
"""Final render characterization: gamma, spatial std, multi-camera, alpha map."""
import os, sys, json
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
import torch, numpy as np
from PIL import Image
dev = torch.device("cuda:6")
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0010000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means = g["gaussians._means"].to(dev).float()
quats = g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = g["gaussians._scales"].to(dev).float().exp()
opac = g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()

cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

def get_gt(c,w2,h2):
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                return torch.tensor(np.asarray(Image.open(pp).convert("RGB").resize((w2,h2),Image.LANCZOS),np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)
    return None

W,H=396,264
print("cam | psnr corr | rend_std gt_std | gamma_corr(2.2) gamma_corr(1/2.2)")
for cam_idx in [0,2,5,8]:
    if cam_idx>=len(cams): break
    c=cams[cam_idx]; scale=W/float(c["width"])
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
    Ks=torch.tensor([[c["fx"]*scale,0,(W-1)/2.0],[0,c["fy"]*scale,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)
    gt=get_gt(c,W,H)
    out=R_fn(means,quats,scales,opac,colors,sh_degree=3,viewmats=vm,Ks=Ks,width=W,height=H,tile_size=16,
             near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=True,use_higs_culling=False)
    rend=out["frame"].permute(0,3,1,2).clamp(0,1)
    mse=(rend-gt).pow(2).mean().item(); psnr=-10*np.log10(max(mse,1e-12))
    corr=np.corrcoef(rend[0].cpu().numpy().ravel(),gt[0].cpu().numpy().ravel())[0,1]
    rstd=rend[0].std().item(); gstd=gt[0].std().item()
    # gamma
    rg=torch.pow(rend.clamp_min(1e-6),2.2); g22=np.corrcoef(rg[0].cpu().numpy().ravel(),gt[0].cpu().numpy().ravel())[0,1]
    rg2=torch.pow(rend.clamp_min(1e-6),1/2.2); g12=np.corrcoef(rg2[0].cpu().numpy().ravel(),gt[0].cpu().numpy().ravel())[0,1]
    print(f"{cam_idx:3d} | {psnr:5.2f} {corr:6.3f} | {rstd:.3f} {gstd:.3f} | {g22:6.3f} {g12:6.3f}")
