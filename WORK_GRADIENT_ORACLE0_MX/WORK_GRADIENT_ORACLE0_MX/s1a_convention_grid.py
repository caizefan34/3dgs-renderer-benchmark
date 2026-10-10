#!/usr/bin/env python3
"""Resolve coordinate convention: test means-transform x camera-transform combos.
nerfidg applies colmap2opengl=diag(1,-1,-1,1) to poses; gaussians may be in that space."""
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
means0 = g["gaussians._means"].to(dev).float()
quats = g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = torch.exp(g["gaussians._scales"].to(dev).float())
opac = torch.sigmoid(g["gaussians._opacities"].to(dev).float()).squeeze(-1)
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(), g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
print(f"means bbox min={means0.min(0).values.tolist()} max={means0.max(0).values.tolist()}")

cams = json.load(open(f"{SCENE}/eval_cameras.json"))
c = cams[0]; scale=0.25
W,H = int(c["width"]*scale), int(c["height"]*scale)
R = np.asarray(c["rotation"],dtype=np.float64); p = np.asarray(c["position"],dtype=np.float64)
Ks = torch.tensor([[float(c["fx"])*scale,0,(W-1)/2.0],[0,float(c["fy"])*scale,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)
C=1; tile=16
gt = Image.open(f"{SCENE}/eval_images/{c['img_name']}.JPG").convert("RGB").resize((W,H),Image.LANCZOS)
gt_np = np.asarray(gt,dtype=np.float32)/255.0
gt_flat = gt_np.ravel()

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

def cam_vm(flip_y=False, flip_z=False):
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    if flip_y: vm[1,:]*=-1; vm[:,1]*=-1
    if flip_z: vm[2,:]*=-1; vm[:,2]*=-1
    return torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)

def transform_means(means, flip_y=False, flip_z=False):
    m = means.clone()
    if flip_y: m[:,1]*=-1
    if flip_z: m[:,2]*=-1
    return m.contiguous()

# Use 1.2M gaussians for speed
sub=min(N,1200000); idx=torch.randperm(N,device=dev)[:sub]
results={}
for mf_y in [False,True]:
    for mf_z in [False,True]:
        m_s = transform_means(means0[idx], mf_y, mf_z)
        for cf_y in [False,True]:
            for cf_z in [False,True]:
                q_s,sc_s,o_s,col_s = quats[idx],scales[idx],opac[idx],colors[idx]
                vm = cam_vm(cf_y, cf_z)
                out = R_fn(m_s,q_s,sc_s,o_s,col_s, sh_degree=sh_degree, viewmats=vm, Ks=Ks,
                           width=W,height=H,tile_size=tile, near_plane=0.2, far_plane=10000.0,
                           background=torch.zeros(3,device=dev), enable_culling=False, use_higs_culling=False)
                frame = out["frame"].detach().squeeze(0).cpu().numpy()
                img = frame.reshape(-1,3).ravel()
                corr = np.corrcoef(img, gt_flat)[0,1]
                mse = ((frame - gt_np)**2).mean()
                key=f"means(y={int(mf_y)},z={int(mf_z)})_cam(y={int(cf_y)},z={int(cf_z)})"
                results[key]={"corr":float(corr),"mse":float(mse)}
                print(f"{key:50s} corr={corr:+.4f} mse={mse:.4f}")

best=max(results, key=lambda k: results[k]["corr"])
print(f"\nBEST: {best}  corr={results[best]['corr']:.4f} mse={results[best]['mse']:.4f}")
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_CONVENTION_GRID.json","w") as f:
    json.dump({"camera":c["img_name"],"best":best,"results":results},f,indent=2)
