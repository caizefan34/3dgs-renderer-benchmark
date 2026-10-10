#!/usr/bin/env python3
"""RENDERER_BACKWARD_CORRECTNESS_GATE.

Decisive diagnostic for the B(gsplat) vs C(HiGS) near-orthogonal dense-gradient finding.
For EACH renderer independently:
  - compute autograd gradient of a fixed scalar loss L w.r.t. 5 param groups
  - compute finite-difference (central difference) gradient of the SAME L
  - compare autograd vs FD (cosine, relL2, per-component) across an epsilon scan
If both renderers' autograd matches their OWN FD -> both backwards correct; B!=C is a
forward-math/convention difference (not a backward bug). If one renderer's autograd !=
its FD -> that renderer has a CUDA backward bug.
Fixed setup: same Gaussian subset, camera, background, SH, loss (MSE, smooth), normalization,
same param indexing/ordering, sparse mask OFF (full render).
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
dev = torch.device("cuda:6"); torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT = f"{D}/gp0_deliverables"; os.makedirs(OUT, exist_ok=True)
import torch.nn.functional as Fn

CKDIR = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
CKPTS = {"5K":"0005000.pt","10K":"0010000.pt","15K":"final.pt"}
cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H=396,264; TILE=16
PK=["means","quats","scales","opac","colors"]
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
from gsplat.rendering import _maybe_evaluate_sh
from gsplat import rendering as _r
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs

def load(ckpt_file):
    blob=torch.load(f"{CKDIR}/{ckpt_file}",map_location="cpu",weights_only=False)
    g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
    means=g["gaussians._means"].to(dev).float()
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales=g["gaussians._scales"].to(dev).float().exp()
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
    return means,quats,scales,opac,colors

def cam_setup(ci):
    c=cams[ci]; sf=W/float(c["width"])
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm4=torch.tensor(vm,dtype=torch.float32,device=dev)
    Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
    return c,sf,vm4,Ks,vm,p

def get_gt(ci):
    c=cams[ci]
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                return np.asarray(Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0
    return None

# ---- renderer forward: returns [1,3,H,W] in [0,1] ----
def render_B(pp, vm1, Ks1):
    N=pp["means"].shape[0]
    radii,means2d,depths,conics,_=fully_fused_projection(means=pp["means"].contiguous(),covars=None,quats=pp["quats"].contiguous(),
        scales=pp["scales"].contiguous(),viewmats=vm1,Ks=Ks1,width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,
        radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
    opac_bc=pp["opac"].unsqueeze(0).contiguous()
    tw=math.ceil(W/TILE); th=math.ceil(H/TILE)
    _,isect_ids,flatten_ids=isect_tiles(means2d,radii,depths,16,tw,th,packed=False,n_images=1,image_ids=None,gaussian_ids=None,conics=conics,opacities=opac_bc)
    isect_offsets=isect_offset_encode(isect_ids,1,tw,th).reshape((1,1,th,tw))
    colors_eval=_maybe_evaluate_sh(3,pp["colors"],pp["means"],radii,vm1,(1,),1,N,True).contiguous()
    bg=torch.zeros((1,3),device=dev); opac_rast=pp["opac"].unsqueeze(0).contiguous()
    fn=_r._make_lazy_cuda_func("rasterize_to_pixels_3dgs")
    rc,ra,_,_=fn(means2d.contiguous(),conics.contiguous(),colors_eval.contiguous(),opac_rast,bg,None,W,H,16,isect_offsets.contiguous(),flatten_ids.contiguous(),False,False)
    return rc.reshape(1,H,W,3).permute(0,3,1,2).clamp(0,1), ra

def render_C(pp, vm1, Ks1):
    out=higs(pp["means"],pp["quats"],pp["scales"],pp["opac"],pp["colors"],sh_degree=3,viewmats=vm1,Ks=Ks1,
             width=W,height=H,tile_size=TILE,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    return out["frame"].permute(0,3,1,2).clamp(0,1), out.get("alpha")

def loss_fn(rgb_chw, gt_t, kind="mse"):
    # IDENTICAL loss for both renderers and FD. kind: 'mse' (smooth) or 'l1' (non-smooth).
    if kind=="mse":
        return ((rgb_chw - gt_t)**2).mean()
    else:
        return (rgb_chw - gt_t).abs().mean()

def autograd_grad(which, base, vm1, Ks1, gt_t, loss_kind="mse"):
    pp={k:base[k].clone().detach().requires_grad_(True) for k in PK}
    rgb,_=render_B(pp,vm1,Ks1) if which=="B" else render_C(pp,vm1,Ks1)
    L=loss_fn(rgb,gt_t,loss_kind)
    g=torch.autograd.grad(L,[pp[k] for k in PK])
    return L.detach().clone(), {PK[i]:g[i].detach().clone() for i in range(5)}

def fd_grad(which, base, vm1, Ks1, gt_t, comps, eps, loss_kind="mse"):
    fd={k:[] for k in PK}
    for (gname, midx) in comps:
        pp_p={k:base[k].clone().detach() for k in PK}
        pp_m={k:base[k].clone().detach() for k in PK}
        pp_p[gname][midx]=pp_p[gname][midx]+eps
        pp_m[gname][midx]=pp_m[gname][midx]-eps
        rp,_=render_B(pp_p,vm1,Ks1) if which=="B" else render_C(pp_p,vm1,Ks1)
        rm,_=render_B(pp_m,vm1,Ks1) if which=="B" else render_C(pp_m,vm1,Ks1)
        Lp=loss_fn(rp,gt_t,loss_kind); Lm=loss_fn(rm,gt_t,loss_kind)
        fd[gname].append(float((Lp-Lm)/(2*eps)))
    return fd

def cosine(a,b):
    a=a.flatten().double(); b=b.flatten().double()
    d=torch.linalg.norm(a)*torch.linalg.norm(b)
    return float(torch.dot(a,b)/d) if d>1e-12 else 0.0
def rell2(a,b):
    a=a.flatten().double(); b=b.flatten().double()
    return float(torch.linalg.norm(a-b)/(torch.linalg.norm(a)+1e-12))

def run(ct, ci, subN=50000, loss_kind="mse"):
    means,quats,scales,opac,colors=load(CKPTS[ct])
    c,sf,vm4,Ks,vm,p=cam_setup(ci)
    gt=get_gt(ci); gt_t=torch.tensor(gt,dtype=torch.float32,device=dev).permute(2,0,1).unsqueeze(0)
    vm1=vm4.unsqueeze(0); Ks1=Ks.unsqueeze(0)
    # on-screen subset
    Rm=vm4[:3,:3]; tm=vm4[:3,3]; cs=(Rm@means.T).T+tm.reshape(1,3)
    z=cs[:,2]; zc=z.clamp_min(1e-6)
    u=(Ks[0,0]*cs[:,0]+Ks[0,2]*zc)/zc; v=(Ks[1,1]*cs[:,1]+Ks[1,2]*zc)/zc
    valid=(z>0.2)&(z<10000)&torch.isfinite(u)&torch.isfinite(v)
    on_screen=valid&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
    idx=on_screen.nonzero(as_tuple=True)[0]
    if idx.numel()>subN:
        sel=torch.randperm(idx.numel(),device=dev)[:subN]; idx=idx[sel]
    idx=idx.sort().values
    base={"means":means[idx].clone().detach(),"quats":quats[idx].clone().detach(),
          "scales":scales[idx].clone().detach(),"opac":opac[idx].clone().detach(),
          "colors":colors[idx].clone().detach()}
    subN=int(idx.numel())
    # FD components: (group, multi_index_tuple). colors is [N,16,3] so needs 3 indices.
    h=subN//2
    comps=[("means",(0,0)),("means",(0,1)),("means",(0,2)),("means",(h,0)),("means",(h,1)),
           ("quats",(0,0)),("quats",(0,1)),("quats",(0,2)),("quats",(0,3)),
           ("scales",(0,0)),("scales",(0,1)),("scales",(0,2)),
           ("opac",(0,)),("opac",(h,)),("opac",(subN-1,)),
           ("colors",(0,0,0)),("colors",(0,0,1)),("colors",(0,0,2)),("colors",(h,0,0))]
    EPS=[1e-1,1e-2,1e-3,1e-4,1e-5]
    res={"ckpt":ct,"cam":ci,"subN":subN,"num_tiles":math.ceil(H/TILE)*math.ceil(W/TILE),"eps_scan":EPS,"loss_kind":loss_kind}
    # autograd for B and C
    LB,gB=autograd_grad("B",base,vm1,Ks1,gt_t,loss_kind)
    LC,gC=autograd_grad("C",base,vm1,Ks1,gt_t,loss_kind)
    res["loss_B"]=float(LB); res["loss_C"]=float(LC); res["loss_absdiff"]=abs(float(LB-LC))
    res["grad_B_vs_C"]={k:{"cosine":cosine(gB[k],gC[k]),"relL2":rell2(gB[k],gC[k]),
                           "B_norm":float(gB[k].norm()),"C_norm":float(gC[k].norm())} for k in PK}
    # FD check per renderer
    for which,g in [("B",gB),("C",gC)]:
        fd_by_eps={}
        for eps in EPS:
            fd=fd_grad(which,base,vm1,Ks1,gt_t,comps,eps,loss_kind)
            # per-group compare autograd vs FD at the checked components
            per_group={}
            for k in PK:
                idxs=[midx for (g2,midx) in comps if g2==k]
                if not idxs: continue
                ad=torch.tensor([float(g[k][midx]) for midx in idxs])
                fdv=torch.tensor(fd[k])
                per_group[k]={"cosine":cosine(ad,fdv),"relL2":rell2(ad,fdv),
                              "autograd":ad.tolist(),"fd":fdv.tolist()}
            fd_by_eps[str(eps)]=per_group
        res[f"fd_{which}"]=fd_by_eps
    return res

if __name__=="__main__":
    ct=sys.argv[1] if len(sys.argv)>1 else "5K"
    ci=int(sys.argv[2]) if len(sys.argv)>2 else 0
    subN=int(sys.argv[3]) if len(sys.argv)>3 else 50000
    loss_kind=sys.argv[4] if len(sys.argv)>4 else "mse"
    print(f"=== gate {ct} cam{ci} subN={subN} loss={loss_kind} ===",flush=True)
    t0=time.time()
    r=run(ct,ci,subN,loss_kind)
    r["seconds"]=time.time()-t0
    # summary: best-eps per group per renderer (min relL2)
    summ={}
    for which in ["B","C"]:
        summ[which]={}
        for k in PK:
            best=None
            for eps in r[f"fd_{which}"]:
                v=r[f"fd_{which}"][eps][k]["relL2"]
                if best is None or v<best[1]: best=(eps,v,r[f"fd_{which}"][eps][k]["cosine"])
            summ[which][k]={"best_eps":best[0],"best_relL2":best[1],"best_cos":best[2]}
    r["summary_best"]=summ
    json.dump(r,open(f"{OUT}/BACKWARD_GATE_{ct}_cam{ci}_{loss_kind}.json","w"),indent=2,default=str)
    print(json.dumps(summ,indent=2))
    print(f"loss_B={r['loss_B']:.6f} loss_C={r['loss_C']:.6f} | gradB_vsC cos means={r['grad_B_vs_C']['means']['cosine']:.4f} scales={r['grad_B_vs_C']['scales']['cosine']:.4f}")
    print(f"saved BACKWARD_GATE_{ct}_cam{ci}_{loss_kind}.json  ({r['seconds']:.1f}s)")
