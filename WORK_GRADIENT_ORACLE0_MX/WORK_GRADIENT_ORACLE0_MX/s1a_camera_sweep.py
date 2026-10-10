#!/usr/bin/env python3
"""Camera convention flip sweep: render ckpt+cam under 4 axis-flip variants,
compare correlation to GT. Definitive convention resolver."""
import os, sys, math, json, time
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
quats = g["gaussians._rotations"].to(dev).float(); quats = quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = torch.exp(g["gaussians._scales"].to(dev).float())
opac = torch.sigmoid(g["gaussians._opacities"].to(dev).float()).squeeze(-1)
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),
                    g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()

cams = json.load(open(f"{SCENE}/eval_cameras.json"))
c = cams[0]; scale=0.25
W,H = int(c["width"]*scale), int(c["height"]*scale)
R = np.asarray(c["rotation"],dtype=np.float64); p = np.asarray(c["position"],dtype=np.float64)
K = np.array([[float(c["fx"])*scale,0,(W-1)/2.0],[0,float(c["fy"])*scale,(H-1)/2.0],[0,0,1.0]],dtype=np.float64)
Ks = torch.tensor(K,dtype=torch.float32,device=dev).unsqueeze(0)
C=1; tile=16; th=math.ceil(H/tile); tw=math.ceil(W/tile)

gt = Image.open(f"{SCENE}/eval_images/{c['img_name']}.JPG").convert("RGB").resize((W,H),Image.LANCZOS)
gt_t = torch.tensor(np.asarray(gt,dtype=np.float32)/255.0,device=dev)  # [H,W,3]
gt_np = gt_t.cpu().numpy().ravel()

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

# Use MORE gaussians for a complete render (reduce subsample artifact)
sub = min(N, 1200000)
idx = torch.randperm(N, device=dev)[:sub]
m_s,q_s,sc_s,o_s,col_s = [t[idx].contiguous() for t in (means,quats,scales,opac,colors)]

def make_viewmat(flip_y=False, flip_z=False, transpose=False):
    Rw2c = R.T if not transpose else R  # if rotation stored as w2c, use directly
    vm = np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3] = -Rw2c @ p
    if flip_y: vm[1,:]*=-1; vm[:,1]*=-1
    if flip_z: vm[2,:]*=-1; vm[:,2]*=-1
    return torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)

variants = {
    "as_is":            dict(flip_y=False,flip_z=False,transpose=False),
    "flip_y":           dict(flip_y=True, flip_z=False,transpose=False),
    "flip_z":           dict(flip_y=False,flip_z=True, transpose=False),
    "flip_yz":          dict(flip_y=True, flip_z=True, transpose=False),
    "rot_is_w2c":       dict(flip_y=False,flip_z=False,transpose=True),
    "rot_w2c_flip_y":   dict(flip_y=True, flip_z=False,transpose=True),
    "rot_w2c_flip_z":   dict(flip_y=False,flip_z=True, transpose=True),
    "rot_w2c_flip_yz":  dict(flip_y=True, flip_z=True, transpose=True),
}

results = {}
for name,cfg in variants.items():
    vm = make_viewmat(**cfg)
    try:
        out = R_fn(m_s,q_s,sc_s,o_s,col_s, sh_degree=sh_degree, viewmats=vm, Ks=Ks,
                   width=W,height=H,tile_size=tile, near_plane=0.2, far_plane=10000.0,
                   background=torch.zeros(3,device=dev), enable_culling=False, use_higs_culling=False)
        frame = out["frame"].detach()  # [1,H,W,3]
        img = frame.squeeze(0).cpu().numpy().ravel()
        corr = np.corrcoef(img, gt_np)[0,1]
        mse = ((frame.squeeze(0)-gt_t).pow(2).mean()).item()
        psnr = -10*np.log10(max(mse,1e-12))
        results[name] = {"corr":float(corr),"psnr":float(psnr),"mse":float(mse)}
        print(f"{name:18s} corr={corr:.4f} psnr={psnr:6.2f}dB mse={mse:.4f}")
    except Exception as e:
        results[name] = {"error":str(e)}
        print(f"{name:18s} FAIL {e}")

best = max((k for k in results if "corr" in results[k]), key=lambda k: results[k]["corr"], default=None)
print(f"\nBEST variant: {best} corr={results[best]['corr']:.4f}")
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_CAMERA_SWEEP.json","w") as f:
    json.dump({"camera":c["img_name"],"N":N,"sub":sub,"render_wh":[W,H],
               "results":results,"best":best}, f, indent=2)
print("saved S1A_CAMERA_SWEEP.json")
