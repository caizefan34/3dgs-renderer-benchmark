#!/usr/bin/env python3
"""Camera convention sweep: find the transform that makes render correlate with GT."""
import os, sys, json, math
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
import torch, numpy as np
from PIL import Image

dev = torch.device("cuda:6")
torch.manual_seed(0)
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means = g["gaussians._means"].to(dev).float()
quats = g["gaussians._rotations"].to(dev).float(); quats = quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = g["gaussians._scales"].to(dev).float().exp()
opac = g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),
                    g["gaussians._sh_coefficients_rest"].to(dev).float()], dim=1).contiguous()

cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W, H = 396, 264
def gt_tensor(c):
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            p=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(p) and os.path.getsize(p)>1000:
                im=Image.open(p).convert("RGB").resize((W,H),Image.LANCZOS)
                return torch.tensor(np.asarray(im,np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)
    return None

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

def build_vm(c, variant):
    scale=W/float(c["width"])
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T
    vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm=torch.tensor(vm,dtype=torch.float32,device=dev)
    if variant=="flip_yz":      # OpenCV<->OpenGL: negate camera y,z
        F=torch.diag(torch.tensor([1,-1,-1,1],dtype=torch.float32,device=dev))
        vm=F@vm
    elif variant=="flip_y":
        F=torch.diag(torch.tensor([1,-1,1,1],dtype=torch.float32,device=dev)); vm=F@vm
    elif variant=="flip_z":
        F=torch.diag(torch.tensor([1,1,-1,1],dtype=torch.float32,device=dev)); vm=F@vm
    elif variant=="R_direct":   # treat R as w2c directly
        vm=np.eye(4); vm[:3,:3]=R; vm[:3,3]=-R@p; vm=torch.tensor(vm,dtype=torch.float32,device=dev)
    elif variant=="flip_all":
        F=torch.diag(torch.tensor([-1,-1,1,1],dtype=torch.float32,device=dev)); vm=F@vm
    return vm

def render_eval(vm, K):
    out=R_fn(means,quats,scales,opac,colors,sh_degree=3,viewmats=vm.unsqueeze(0),Ks=K.unsqueeze(0),
             width=W,height=H,tile_size=16,near_plane=0.01,far_plane=1e10,
             background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    return out["frame"]  # [1,H,W,3]

for cam_idx in [0, 2]:
    c=cams[cam_idx]; scale=W/float(c["width"])
    K=torch.tensor([[float(c["fx"])*scale,0,(W-1)/2.0],[0,float(c["fy"])*scale,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
    gt=gt_tensor(c)
    print(f"\n=== camera {cam_idx} {c['img_name']} ===")
    for variant in ["base","flip_yz","flip_y","flip_z","R_direct","flip_all"]:
        vm=build_vm(c,variant)
        rend=render_eval(vm,K).permute(0,3,1,2).clamp(0,1)
        mse=(rend-gt).pow(2).mean().item(); psnr=-10*np.log10(max(mse,1e-12))
        corr=np.corrcoef(rend[0].cpu().numpy().ravel(),gt[0].cpu().numpy().ravel())[0,1]
        print(f"  {variant:10s} PSNR={psnr:6.2f} corr={corr:.3f} rend_mean={rend.mean():.3f}")
