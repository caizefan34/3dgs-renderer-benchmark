#!/usr/bin/env python3
"""Isolate path B (gsplat low-level) full-set render crash."""
import os,sys,json,math
import numpy as np, torch
BUILD_DIR="/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0,BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"]="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
dev=torch.device("cuda:6")
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
from gsplat.rendering import _maybe_evaluate_sh
from gsplat import rendering as _r
CKPT="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob=torch.load(CKPT,map_location="cpu",weights_only=False)
g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means=g["gaussians._means"].to(dev).float()
quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales=g["gaussians._scales"].to(dev).float().exp()
opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
print("N=",means.shape[0],flush=True)
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c=cams[0]; W,H=396,264; sf=W/4946.0
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
vm4=torch.tensor(vm,dtype=torch.float32,device=dev)
Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
print("calling fully_fused_projection...",flush=True)
radii,means2d,depths,conics,_=fully_fused_projection(means=means.contiguous(),covars=None,quats=quats.contiguous(),
    scales=scales.contiguous(),viewmats=vm4.unsqueeze(0),Ks=Ks.unsqueeze(0),width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,
    radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
torch.cuda.synchronize()
print("projection OK radii sum=",int(radii.sum()),flush=True)
