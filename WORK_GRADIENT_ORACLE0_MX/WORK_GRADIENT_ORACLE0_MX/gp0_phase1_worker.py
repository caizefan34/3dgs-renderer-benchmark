#!/usr/bin/env python3
"""GRADIENT-PROXY-P0 Phase 1 worker: renders one (ckpt, path) and computes dense grad.
Run as subprocess to avoid CUDA state corruption across paths."""
import os, sys, json, math
import numpy as np, torch
import torch.nn.functional as Fn
from PIL import Image
BUILD_DIR="/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0,BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"]="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME","/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
dev=torch.device("cuda:6")
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
from gsplat.rendering import _maybe_evaluate_sh
from gsplat import rendering as _r
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs

CKDIR="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
CKPTS={"5K":"0005000.pt","10K":"0010000.pt","15K":"final.pt"}
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H=396,264
PK=["means","quats","scales","opac","colors"]

def get_gt(ci,W,H):
    c=cams[ci]
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                return np.asarray(Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0, pp
    return None,None

def cam_setup(ci,W,H):
    c=cams[ci]; sf=W/float(c["width"])
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm4=torch.tensor(vm,dtype=torch.float32,device=dev)
    Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
    return c,sf,vm4,Ks,vm,p

def load(ckpt_file):
    blob=torch.load(f"{CKDIR}/{ckpt_file}",map_location="cpu",weights_only=False)
    g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
    means=g["gaussians._means"].to(dev).float()
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales=g["gaussians._scales"].to(dev).float().exp()
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
    return means,quats,scales,opac,colors,blob.get("num_iterations_trained")

which=sys.argv[1]; ct=sys.argv[2]; ci=int(sys.argv[3]); OUTD=sys.argv[4]; mode=sys.argv[5] if len(sys.argv)>5 else "render"
means,quats,scales,opac,colors,num_iters=load(CKPTS[ct])
c,sf,vm4,Ks,vm,p=cam_setup(ci,W,H)
gt,gt_path=get_gt(ci,W,H)
N=means.shape[0]
# visible vs total
R=vm4[:3,:3].clone(); t=vm4[:3,3].clone()
cp=(R@means.T).T+t.reshape(1,3)
z=cp[:,2].clone()
# pixel coords only valid for z>0
zc=z.clamp_min(1e-6)
u=(Ks[0,0]*cp[:,0]+Ks[0,2]*zc)/zc
v=(Ks[1,1]*cp[:,1]+Ks[1,2]*zc)/zc
valid=(z>0.2)&(z<10000)&torch.isfinite(u)&torch.isfinite(v)
on_screen=valid&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
vis_op=on_screen&(opac>0.01)
print(f"[{ct} cam{ci}] z[min/med/max]={z.min():.2f}/{z.median():.2f}/{z.max():.2f} u[min/max]={u.min():.1f}/{u.max():.1f} v[min/max]={v.min():.1f}/{v.max():.1f} valid={int(valid.sum())} on_screen={int(on_screen.sum())}",flush=True)
meta={"which":which,"ckpt":ct,"cam":ci,"N_total":int(N),"num_iters":int(num_iters) if num_iters else None,
      "in_frustum":int(valid.sum()),"on_screen":int(on_screen.sum()),"visible_opac":int(vis_op.sum()),
      "opacity_mean":float(opac.mean()),"opacity_median":float(opac.median()),
      "cam_id":c["id"],"img_name":c["img_name"],"fx":float(c["fx"]*sf),"fy":float(c["fy"]*sf),
      "W":W,"H":H,"scale":sf,"gt_path":gt_path}

def path_B(pp):
    radii,means2d,depths,conics,_=fully_fused_projection(means=pp["means"].contiguous(),covars=None,quats=pp["quats"].contiguous(),
        scales=pp["scales"].contiguous(),viewmats=vm4.unsqueeze(0),Ks=Ks.unsqueeze(0),width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,
        radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
    opac_bc=pp["opac"].unsqueeze(0).contiguous()
    tw=math.ceil(W/16); th=math.ceil(H/16)
    _,isect_ids,flatten_ids=isect_tiles(means2d,radii,depths,16,tw,th,packed=False,n_images=1,image_ids=None,gaussian_ids=None,conics=conics,opacities=opac_bc)
    isect_offsets=isect_offset_encode(isect_ids,1,tw,th).reshape((1,1,th,tw))
    colors_eval=_maybe_evaluate_sh(3,pp["colors"],pp["means"],radii,vm4.unsqueeze(0),(1,),1,pp["means"].shape[0],True).contiguous()
    bg=torch.zeros((1,3),device=dev); opac_rast=pp["opac"].unsqueeze(0).contiguous()
    fn=_r._make_lazy_cuda_func("rasterize_to_pixels_3dgs")
    rc,ra,_,_=fn(means2d.contiguous(),conics.contiguous(),colors_eval.contiguous(),opac_rast,bg,None,W,H,16,isect_offsets.contiguous(),flatten_ids.contiguous(),False,False)
    return rc.reshape(1,H,W,3).permute(0,3,1,2).clamp(0,1), ra, radii

def path_C(pp):
    out=higs(pp["means"],pp["quats"],pp["scales"],pp["opac"],pp["colors"],sh_degree=3,viewmats=vm4.unsqueeze(0),Ks=Ks.unsqueeze(0),
             width=W,height=H,tile_size=16,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    return out["frame"].permute(0,3,1,2).clamp(0,1), out.get("metadata"), out.get("alpha")

gt_t=torch.tensor(gt,dtype=torch.float32,device=dev).permute(2,0,1).unsqueeze(0)

if mode=="render":
    # full-set render
    pp={k:v.clone().detach() for k,v in zip(PK,[means,quats,scales,opac,colors])}
    if which=="B":
        rgb,alpha,radii=path_B(pp); meta["alpha_mean"]=float(alpha.mean()) if alpha is not None else None
        arr=rgb[0].detach().cpu().numpy().transpose(1,2,0)
    else:
        rgb,mC,alpha=path_C(pp); meta["higs_meta"]={k:(v.item() if isinstance(v,torch.Tensor) and v.numel()==1 else str(v)) for k,v in mC.items()} if mC else {}
        meta["alpha_mean"]=float(alpha.mean()) if alpha is not None else None
        arr=rgb[0].detach().cpu().numpy().transpose(1,2,0)
    mse=np.square(arr.astype(np.float64)-gt.astype(np.float64)).mean()
    meta["psnr_vs_gt"]=float(-10*np.log10(max(mse,1e-12))); meta["mae_vs_gt"]=float(np.abs(arr.astype(np.float64)-gt.astype(np.float64)).mean())
    meta["arr_mean"]=float(arr.mean()); meta["arr_std"]=float(arr.std())
    np.save(f"{OUTD}/{ct}_{which}_cam{ci}.npy", arr.astype(np.float32))
    Image.fromarray((arr*255).astype(np.uint8)).save(f"{OUTD}/{ct}_{which}_cam{ci}.png")
    print("META:"+json.dumps(meta))
elif mode=="grad":
    # dense gradient on visible subset (200K)
    sub=vis_op.nonzero(as_tuple=True)[0]
    if sub.numel()>200000:
        sel=torch.randperm(sub.numel(),device=dev)[:200000]; sub=sub[sel]
    sub=sub.sort().values
    def mk():
        return {"means":means[sub].clone().detach().requires_grad_(True),"quats":quats[sub].clone().detach().requires_grad_(True),
                "scales":scales[sub].clone().detach().requires_grad_(True),"opac":opac[sub].clone().detach().requires_grad_(True),
                "colors":colors[sub].clone().detach().requires_grad_(True)}
    pp=mk()
    if which=="B": rgb,_,_=path_B(pp)
    else: rgb,_,_=path_C(pp)
    loss=Fn.l1_loss(rgb[0],gt_t[0])
    g=torch.autograd.grad(loss,[pp[k] for k in PK])
    grads={PK[i]:g[i].detach().cpu().numpy() for i in range(len(PK))}
    np.savez(f"{OUTD}/{ct}_{which}_cam{ci}_grad.npz", **grads)
    meta["grad_sub_N"]=int(sub.numel()); meta["loss"]=float(loss)
    print("META:"+json.dumps(meta))
