#!/usr/bin/env python3
"""Save render variants as uint8 PNGs for visual diagnosis."""
import os, sys, math, json
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("PYTHONNOUSERSITE","1")
import torch, numpy as np
from PIL import Image
dev = "cuda:6"
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
SCENE = "/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
N = g["gaussians._means"].shape[0]; sh_degree=3
means = g["gaussians._means"].to(dev).float()
quats = g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = torch.exp(g["gaussians._scales"].to(dev).float())
opac = torch.sigmoid(g["gaussians._opacities"].to(dev).float()).squeeze(-1)
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(), g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
cams = json.load(open(f"{SCENE}/eval_cameras.json"))
c = cams[0]; scale=0.25
W,H = int(c["width"]*scale), int(c["height"]*scale)
R = np.asarray(c["rotation"],dtype=np.float64); p = np.asarray(c["position"],dtype=np.float64)
Ks = torch.tensor([[float(c["fx"])*scale,0,(W-1)/2.0],[0,float(c["fy"])*scale,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)
C=1; tile=16
gt = Image.open(f"{SCENE}/eval_images/{c['img_name']}.JPG").convert("RGB").resize((W,H),Image.LANCZOS)
gt_np = np.asarray(gt,dtype=np.uint8)

# FULL gaussians at this res for a complete render
m_s,q_s,sc_s,o_s,col_s = means,quats,scales,opac,colors
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

def render(vm):
    out = R_fn(m_s,q_s,sc_s,o_s,col_s, sh_degree=sh_degree, viewmats=vm, Ks=Ks,
               width=W,height=H,tile_size=tile, near_plane=0.2, far_plane=10000.0,
               background=torch.zeros(3,device=dev), enable_culling=False, use_higs_culling=False)
    frame = out["frame"].detach().squeeze(0)  # [H,W,3]
    return (frame.clamp(0,1).cpu().numpy()*255).astype(np.uint8), out.get('metadata',{}).get('n_isects')

def vm_as_is():
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    return torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
def vm_flipz():
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm[2,:]*=-1; vm[:,2]*=-1
    return torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)

r_as_is, ni1 = render(vm_as_is())
r_flipz, ni2 = render(vm_flipz())
print(f"as_is n_isects={ni1}  flipz n_isects={ni2}")
# side by side: GT | as_is | flipz
combo = np.concatenate([gt_np, r_as_is, r_flipz], axis=1)
Image.fromarray(combo).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/sanity_sbs.png")
print("saved sanity_sbs.png (GT | as_is | flip_z)")
