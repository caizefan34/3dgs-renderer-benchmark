#!/usr/bin/env python3
"""Resolve the contradiction: gate gives cos=1.0, saved Phase 1 npz gives cos=0.005.
Test same-process sequential B then C, and opac-filter subset, vs separate processes."""
import os, sys, json, math, time
import numpy as np, torch
import torch.nn.functional as Fn
BUILD_DIR="/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0,BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"]="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
dev=torch.device("cuda:6"); torch.manual_seed(0)
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
from gsplat.rendering import _maybe_evaluate_sh
from gsplat import rendering as _r
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs
from PIL import Image
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"; OUT=f"{D}/gp0_deliverables"
CKDIR="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H=396,264; TILE=16; PK=["means","quats","scales","opac","colors"]
def load(ct):
    blob=torch.load(f"{CKDIR}/{'0005000.pt' if ct=='5K' else '0010000.pt' if ct=='10K' else 'final.pt'}",map_location="cpu",weights_only=False)
    g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
    means=g["gaussians._means"].to(dev).float()
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales=g["gaussians._scales"].to(dev).float().exp()
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
    return means,quats,scales,opac,colors
def cam_setup(ci):
    c=cams[ci]; sf=W/float(c["width"])
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64); Rw2c=R.T
    vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm4=torch.tensor(vm,dtype=torch.float32,device=dev)
    Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
    return vm4,Ks
def get_gt(ci):
    c=cams[ci]
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                return np.asarray(Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0
    return None
def render_B(pp,vm1,Ks1):
    N=pp["means"].shape[0]
    radii,means2d,depths,conics,_=fully_fused_projection(means=pp["means"].contiguous(),covars=None,quats=pp["quats"].contiguous(),scales=pp["scales"].contiguous(),viewmats=vm1,Ks=Ks1,width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
    opac_bc=pp["opac"].unsqueeze(0).contiguous()
    tw=math.ceil(W/TILE); th=math.ceil(H/TILE)
    _,isect_ids,flatten_ids=isect_tiles(means2d,radii,depths,16,tw,th,packed=False,n_images=1,image_ids=None,gaussian_ids=None,conics=conics,opacities=opac_bc)
    isect_offsets=isect_offset_encode(isect_ids,1,tw,th).reshape((1,1,th,tw))
    colors_eval=_maybe_evaluate_sh(3,pp["colors"],pp["means"],radii,vm1,(1,),1,N,True).contiguous()
    bg=torch.zeros((1,3),device=dev); opac_rast=pp["opac"].unsqueeze(0).contiguous()
    fn=_r._make_lazy_cuda_func("rasterize_to_pixels_3dgs")
    rc,ra,_,_=fn(means2d.contiguous(),conics.contiguous(),colors_eval.contiguous(),opac_rast,bg,None,W,H,16,isect_offsets.contiguous(),flatten_ids.contiguous(),False,False)
    return rc.reshape(1,H,W,3).permute(0,3,1,2).clamp(0,1),ra
def render_C(pp,vm1,Ks1):
    out=higs(pp["means"],pp["quats"],pp["scales"],pp["opac"],pp["colors"],sh_degree=3,viewmats=vm1,Ks=Ks1,width=W,height=H,tile_size=TILE,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    return out["frame"].permute(0,3,1,2).clamp(0,1),out.get("alpha")
def loss_fn(rgb,gt): return ((rgb-gt)**2).mean()
def cos(a,b):
    a=np.asarray(a).flatten(); b=np.asarray(b).flatten()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def autograd(which,base,vm1,Ks1,gt_t):
    pp={k:base[k].clone().detach().requires_grad_(True) for k in PK}
    rgb,_=render_B(pp,vm1,Ks1) if which=="B" else render_C(pp,vm1,Ks1)
    L=loss_fn(rgb,gt_t)
    g=torch.autograd.grad(L,[pp[k] for k in PK])
    return L.detach().clone(),{PK[i]:g[i].detach().clone() for i in range(5)}

ct="5K"; ci=0; subN=200000
means,quats,scales,opac,colors=load(ct)
vm4,Ks=cam_setup(ci); gt_t=torch.tensor(get_gt(ci),dtype=torch.float32,device=dev).permute(2,0,1).unsqueeze(0)
vm1=vm4.unsqueeze(0); Ks1=Ks.unsqueeze(0)
Rm=vm4[:3,:3]; tm=vm4[:3,3]; cs=(Rm@means.T).T+tm.reshape(1,3)
z=cs[:,2]; zc=z.clamp_min(1e-6)
u=(Ks[0,0]*cs[:,0]+Ks[0,2]*zc)/zc; v=(Ks[1,1]*cs[:,1]+Ks[1,2]*zc)/zc
valid=(z>0.2)&(z<10000)&torch.isfinite(u)&torch.isfinite(v)
on_screen=valid&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
vis_op=on_screen&(opac>0.01)
def subset(mask):
    idx=mask.nonzero(as_tuple=True)[0]
    if idx.numel()>subN: idx=idx[:subN]
    idx=idx.sort().values
    return {"means":means[idx].clone().detach(),"quats":quats[idx].clone().detach(),"scales":scales[idx].clone().detach(),"opac":opac[idx].clone().detach(),"colors":colors[idx].clone().detach()},int(idx.numel())
res={}
for label,mask in [("on_screen",on_screen),("vis_op",vis_op)]:
    base,n=subset(mask)
    # SAME-PROCESS sequential: B then C
    LB,gB=autograd("B",base,vm1,Ks1,gt_t)
    torch.cuda.synchronize()
    LC,gC=autograd("C",base,vm1,Ks1,gt_t)
    torch.cuda.synchronize()
    res[label]={"n":n,"same_proc_seq_B_then_C":{k:cos(gB[k].cpu().numpy(),gC[k].cpu().numpy()) for k in PK},
                "LB":float(LB),"LC":float(LC)}
    print(f"[{label} n={n} SAME-PROC SEQ] cos:",res[label]["same_proc_seq_B_then_C"],f"LB={LB:.6f} LC={LC:.6f}",flush=True)
json.dump(res,open(f"{OUT}/BACKWARD_GATE_CONTRADICTION.json","w"),indent=2,default=str)
print("saved BACKWARD_GATE_CONTRADICTION.json")
