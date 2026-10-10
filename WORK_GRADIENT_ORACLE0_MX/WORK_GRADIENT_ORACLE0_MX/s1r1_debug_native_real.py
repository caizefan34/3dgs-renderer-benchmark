#!/usr/bin/env python3
"""Debug native FasterGS render on REAL 5K gaussians, isolated."""
import os, sys, math, json
import numpy as np
import torch
_C_so_dir = "/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting/FasterGSFusedCudaBackend/FasterGSFusedCudaBackend"
sys.path.insert(0, _C_so_dir)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
import _C
dev = torch.device("cuda:6")

CKPT="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob=torch.load(CKPT,map_location="cpu",weights_only=False)
g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means=g["gaussians._means"].to(dev).float()
quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales=g["gaussians._scales"].to(dev).float().exp()
opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
sh0=g["gaussians._sh_coefficients_0"].to(dev).float()
shrest=g["gaussians._sh_coefficients_rest"].to(dev).float()
print("loaded N=",means.shape[0], "sh0",sh0.shape,"shrest",shrest.shape,"opac",opac.shape)

cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c=cams[0]
W,H=396,264; sf=W/4946.0
import numpy as np
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
w2c=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
cam_position=torch.tensor(p,dtype=torch.float32,device=dev).reshape(1,3)
bg=torch.zeros(3,device=dev)

# try active_sh_bases = 0 (DC only) first, then 3
for ash in [0,3]:
    try:
        res=_C.forward(means,scales,quats,opac,sh0,shrest,bg,
                       w2c,cam_position,ash,W,H,c["fx"]*sf,c["fy"]*sf,(W-1)/2.0,(H-1)/2.0,0.2,10000.0,0.0,0)
        img=res[0]
        torch.cuda.synchronize()
        print(f"ash={ash} OK img={tuple(img.shape)} min={img.min():.3f} max={img.max():.3f} mean={img.mean():.3f} counts={res[5:]}")
        break
    except Exception as e:
        torch.cuda.synchronize()
        print(f"ash={ash} FAILED: {type(e).__name__}: {e}")
        break
