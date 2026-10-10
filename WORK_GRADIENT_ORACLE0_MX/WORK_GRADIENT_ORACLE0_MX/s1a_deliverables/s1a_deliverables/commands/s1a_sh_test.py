#!/usr/bin/env python3
"""Test sh_degree=0 vs 3, SSIM, and 10K to isolate the render issue."""
import os, sys, json
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
import torch, numpy as np
from PIL import Image
from torch.nn.functional import conv2d
dev = torch.device("cuda:6"); torch.manual_seed(0)

CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means = g["gaussians._means"].to(dev).float()
quats = g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = g["gaussians._scales"].to(dev).float().exp()
opac = g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()

cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H = 396,264
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

def ssim(x, y):
    # x,y: [1,3,H,W] in [0,1]; channel-mean SSIM
    C1,C2 = 0.01**2, 0.03**2
    win = torch.ones(1,1,11,11,device=dev)/121
    win = win.expand(3,1,11,11).contiguous()
    mu_x = conv2d(x, win, padding=5, groups=3)
    mu_y = conv2d(y, win, padding=5, groups=3)
    sx = conv2d(x*x, win, padding=5, groups=3) - mu_x*mu_x
    sy = conv2d(y*y, win, padding=5, groups=3) - mu_y*mu_y
    sxy = conv2d(x*y, win, padding=5, groups=3) - mu_x*mu_y
    return (((2*mu_x*mu_y+C1)*(2*sxy+C2))/((mu_x*mu_x+mu_y*mu_y+C1)*(sx+sy+C2))).mean().item()

c = cams[0]; scale=W/float(c["width"])
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
vm=torch.tensor(vm,dtype=torch.float32,device=dev)
K=torch.tensor([[c["fx"]*scale,0,(W-1)/2.0],[0,c["fy"]*scale,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)

# GT
gt=None
for sub in ("eval_images","images"):
    for ext in (".JPG",".jpg",".png"):
        pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
        if os.path.exists(pp) and os.path.getsize(pp)>1000:
            gt=Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS); break
    if gt is not None: break
gt_t=torch.tensor(np.asarray(gt,np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)

for shd in [0, 1, 3]:
    out=R_fn(means,quats,scales,opac,colors,sh_degree=shd,viewmats=vm.unsqueeze(0),Ks=K.unsqueeze(0),
             width=W,height=H,tile_size=16,near_plane=0.01,far_plane=1e10,
             background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    rend=out["frame"].permute(0,3,1,2).clamp(0,1)
    mse=(rend-gt_t).pow(2).mean().item(); psnr=-10*np.log10(max(mse,1e-12))
    corr=np.corrcoef(rend[0].cpu().numpy().ravel(),gt_t[0].cpu().numpy().ravel())[0,1]
    s=ssim(rend,gt_t)
    print(f"sh_degree={shd}: PSNR={psnr:.2f} corr={corr:.3f} SSIM={s:.3f} rend_mean={rend.mean():.3f}")
    if shd==0:
        Image.fromarray((rend[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_render_sh0.png")

# Also: render with culling enabled (as training might use) and larger tile
for ts in [16, 32]:
    out=R_fn(means,quats,scales,opac,colors,sh_degree=0,viewmats=vm.unsqueeze(0),Ks=K.unsqueeze(0),
             width=W,height=H,tile_size=ts,near_plane=0.2,far_plane=10000.0,
             background=torch.zeros(3,device=dev),enable_culling=True,use_higs_culling=False)
    rend=out["frame"].permute(0,3,1,2).clamp(0,1)
    mse=(rend-gt_t).pow(2).mean().item(); psnr=-10*np.log10(max(mse,1e-12))
    corr=np.corrcoef(rend[0].cpu().numpy().ravel(),gt_t[0].cpu().numpy().ravel())[0,1]
    print(f"sh0 tile={ts} near=0.2 cull=True: PSNR={psnr:.2f} corr={corr:.3f}")
print("saved s1a_render_sh0.png")
