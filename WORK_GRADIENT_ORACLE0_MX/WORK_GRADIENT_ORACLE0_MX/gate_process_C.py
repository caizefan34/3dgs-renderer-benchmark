import os, sys, math, json
import numpy as np, torch
import torch.nn.functional as Fn
BUILD_DIR="/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0,BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"]="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
dev=torch.device("cuda:6")
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs
from PIL import Image
CKDIR="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H=396,264; PK=["means","quats","scales","opac","colors"]
blob=torch.load(f"{CKDIR}/0005000.pt",map_location="cpu",weights_only=False)
g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means=g["gaussians._means"].to(dev).float()
quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales=g["gaussians._scales"].to(dev).float().exp()
opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
c=cams[0]; sf=W/float(c["width"])
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64); Rw2c=R.T
vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
vm4=torch.tensor(vm,dtype=torch.float32,device=dev)
Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
gt=None
for sub in ("eval_images","images"):
    for ext in (".JPG",".jpg",".png"):
        pp="/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/"+sub+"/"+c["img_name"]+ext
        if os.path.exists(pp) and os.path.getsize(pp)>1000:
            gt=np.asarray(Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0; break
    if gt is not None: break
gt_t=torch.tensor(gt,dtype=torch.float32,device=dev).permute(2,0,1).unsqueeze(0)
vm1=vm4.unsqueeze(0); Ks1=Ks.unsqueeze(0)
Rm=vm4[:3,:3]; tm=vm4[:3,3]; cs=(Rm@means.T).T+tm.reshape(1,3)
z=cs[:,2]; zc=z.clamp_min(1e-6)
u=(Ks[0,0]*cs[:,0]+Ks[0,2]*zc)/zc; v=(Ks[1,1]*cs[:,1]+Ks[1,2]*zc)/zc
valid=(z>0.2)&(z<10000)&torch.isfinite(u)&torch.isfinite(v)
on_screen=valid&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
vis_op=on_screen&(opac>0.01)
sub=vis_op.nonzero(as_tuple=True)[0]
torch.manual_seed(0)
sel=torch.randperm(sub.numel(),device=dev)[:200000]; sub=sub[sel]
sub=sub.sort().values
pp={k:v[sub].clone().detach().requires_grad_(True) for k,v in zip(PK,[means,quats,scales,opac,colors])}
out=higs(pp["means"],pp["quats"],pp["scales"],pp["opac"],pp["colors"],sh_degree=3,viewmats=vm1,Ks=Ks1,width=W,height=H,tile_size=16,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
rgb=out["frame"].permute(0,3,1,2).clamp(0,1)
L=Fn.l1_loss(rgb[0],gt_t[0])
gg=torch.autograd.grad(L,[pp[k] for k in PK])
arr=rgb[0].detach().cpu().numpy().transpose(1,2,0)
np.save(f"/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/gp0_renders/proc_C_render.npy",arr.astype(np.float32))
np.savez(f"/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/gp0_renders/proc_C_grad.npz", **{PK[i]:gg[i].detach().cpu().numpy() for i in range(5)})
print("PROC_C loss=%.8f render_sum=%.6f"%(float(L),float(arr.sum())))
