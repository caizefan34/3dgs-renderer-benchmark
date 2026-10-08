#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PBR0 benchmark: per-camera decomposed timing (M1-M4) for impls A/B/C,
resolution native/down2, plus TFC1(B) vs TFC2(C) membership + rendered-output reconciliation.
EPIC-05 A100 only. Measurement only; no algorithm changes."""
import sys, os, time, json, math, argparse
sys.path.insert(0, "/root/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/root/tfc2/kernel")
sys.path.insert(0, "/root/tfc0/kernel")
import numpy as np, torch
import tfc1_rowwise as B
import tfc2_ablation as C
import tfc0_lib as L
from gsplat import (fully_fused_projection, isect_offset_encode,
                    rasterize_to_pixels, spherical_harmonics, isect_tiles)
from benchmark_framework.scene import load_ply
from PIL import Image

dev = "cuda"
TILE = L.TILE; EPS2D = L.EPS2D; NEAR = L.NEAR; FAR = L.FAR; RC = 0.0
REPO = "/root/3dgs-renderer-benchmark"

def colmap_dir(s): return os.path.join(REPO,"data","datasets","mipnerf360",s,"sparse","0")
def ply_official(s): return os.path.join(REPO,"data","official","mipnerf360",s,"point_cloud.ply")
def gt_path(scene, name):
    base = os.path.join(REPO,"data","datasets","mipnerf360",scene,"images")
    if os.path.splitext(name)[1].lower() in ('.jpg','.png','.jpeg'):
        p = os.path.join(base, name)
        if os.path.exists(p): return p
    for ext in ('.JPG','.jpg','.PNG','.png','.jpeg'):
        p = os.path.join(base, name+ext)
        if os.path.exists(p): return p
    return None
def load_gt_tensor(path, W, H):
    img = Image.open(path).convert("RGB")
    if img.size[0] != W or img.size[1] != H:
        img = img.resize((W,H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)/255.0
    return torch.from_numpy(arr).to(dev)

def arr(v):
    v = v.numpy() if hasattr(v,"numpy") else np.asarray(v)
    return v.astype(np.float32)

def load_scene(scene):
    cd = colmap_dir(scene)
    data = load_ply(ply_official(scene), device="cpu")
    xyz=arr(data["xyz"]); scales=np.exp(arr(data["scales"])); quats=arr(data["rotations"])
    opac=arr(data["opacity"]); shs=arr(data["shs"])
    SH=int(round(math.sqrt(shs.shape[1])-1))
    imgs=L.read_images_bin(os.path.join(cd,"images.bin")); cams=L.read_cameras_bin(os.path.join(cd,"cameras.bin"))
    return dict(scene=scene,N=int(xyz.shape[0]),SH=SH,W=cams["W"],H=cams["H"],ids=sorted(imgs.keys()),imgs=imgs,cams=cams,
                xyz=xyz,scales=scales,quats=quats,opac=opac,shs=shs)

def to_gpu(sc):  # returns (means, quats, scales, opac, shs) matching project() signature
    return (torch.from_numpy(sc["xyz"]).to(dev), torch.from_numpy(sc["quats"]).to(dev),
            torch.from_numpy(sc["scales"]).to(dev), torch.from_numpy(sc["opac"]).to(dev),
            torch.from_numpy(sc["shs"]).to(dev))

def cam_tensors(sc, ci, W, H):
    imgs,cams = sc["imgs"],sc["cams"]
    fx,fy,cx,cy = cams["params"][:4]
    fx=fx*W/sc["W"]; fy=fy*H/sc["H"]; cx=cx*W/sc["W"]; cy=cy*H/sc["H"]  # scale intrinsics to resolution
    R=L.qvec2rot(imgs[ci]["qvec"]); t=np.array(imgs[ci]["tvec"])
    viewmat=np.eye(4); viewmat[:3,:3]=R; viewmat[:3,3]=t
    c2w=np.linalg.inv(viewmat); cam_center=c2w[:3,3]
    viewmat_t=torch.tensor(viewmat,dtype=torch.float32,device=dev).unsqueeze(0)
    K_t=torch.tensor(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32),device=dev).unsqueeze(0)
    cam_center_t=torch.tensor(cam_center,dtype=torch.float32,device=dev)
    return viewmat_t,K_t,cam_center_t,imgs[ci]["name"]

def ev(fn, iters=5, warmup=2):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best=1e9; nisect=0
    for _ in range(iters):
        torch.cuda.synchronize(); t0=time.time(); r=fn(); torch.cuda.synchronize(); best=min(best,time.time()-t0)
    return best, r

def project(means,quats,scales,opac,viewmat,K,W,H,sh_degree):
    bid,cid,gids,radii,means2d,depths,conics,comp = fully_fused_projection(
        means=means,covars=None,quats=quats,scales=scales,viewmats=viewmat,Ks=K,
        width=W,height=H,eps2d=EPS2D,near_plane=NEAR,far_plane=FAR,radius_clip=RC,
        packed=True,opacities=torch.sigmoid(opac),calc_compensations=False)
    return gids,radii,means2d,depths,conics

def isect_A(means2d,radii,depths,cid,gids,W,H):
    TW=(W+TILE-1)//TILE; TH=(H+TILE-1)//TILE
    return isect_tiles(means2d,radii.to(torch.int32),depths,TILE,TW,TH,sort=True,packed=True,image_ids=cid,gaussian_ids=gids,n_images=1)
def isect_BC(mod,means2d,conics,opac,depths,W,H,mode):
    return mod.footprint_intersect(means2d,conics,opac,depths,None,float(TILE),W,H,RC,mode)

def raster(means2d,conics,colors,opac_render,W,H,offsets,flat_ids):
    r,_=rasterize_to_pixels(means2d,conics,colors,opac_render,W,H,TILE,offsets,flat_ids,packed=True)
    return r

def measure_camera(sc,ci,W,H,impl,means,quats,scales,opac,shs,sh_degree,target,opt=None):
    viewmat,K,cam_center,name = cam_tensors(sc,ci,W,H)
    TW=(W+TILE-1)//TILE; TH=(H+TILE-1)//TILE
    res={}
    # --- M2 forward pipeline (decomposed), per impl ---
    def fwd_pipeline():
        gids,radii,means2d,depths,conics = project(means,quats,scales,opac,viewmat,K,W,H,sh_degree)
        if impl=="A":
            tpg,isect_ids,flat = isect_A(means2d,radii,depths,torch.zeros_like(gids),gids,W,H)
        elif impl=="B":
            tpg,isect_ids,flat = isect_BC(B,means2d,conics,torch.sigmoid(opac)[gids],depths,W,H,1)
        else:
            tpg,isect_ids,flat = isect_BC(C,means2d,conics,torch.sigmoid(opac)[gids],depths,W,H,1)
        offsets=isect_offset_encode(isect_ids,1,TW,TH)
        dirs=means[gids].detach().clone()-cam_center.unsqueeze(0)
        col=torch.clamp_min(spherical_harmonics(sh_degree,dirs,shs[gids])+0.5,0.0).contiguous()
        orr=torch.sigmoid(opac[gids]).contiguous()
        r=raster(means2d,conics,col,orr,W,H,offsets,flat)
        return r,isect_ids,flat,means2d,conics
    t_fwd,(render,isect_ids,flat,means2d,conics) = ev(fwd_pipeline)
    res["m2_forward_pipeline_s"]=t_fwd
    res["isect_count_nnz"]=int(isect_ids.shape[0])
    res["render_shape"]=list(render.shape)
    # --- M1 intersection/sort workload: time isect stage alone ---
    gids,radii,means2d,depths,conics = project(means,quats,scales,opac,viewmat,K,W,H,sh_degree)
    def isect_only():
        if impl=="A":
            return isect_A(means2d,radii,depths,torch.zeros_like(gids),gids,W,H)
        return isect_BC(B if impl=="B" else C,means2d,conics,torch.sigmoid(opac)[gids],depths,W,H,1)
    t_isect,_=ev(isect_only)
    res["m1_isect_s"]=t_isect
    res["m1_sort_input_size"]=int(isect_ids.shape[0])
    # --- M3 forward + backward (autograd through raster) ---
    m2=means.clone().detach().requires_grad_(True); q2=quats.clone().detach().requires_grad_(True)
    s2=scales.clone().detach().requires_grad_(True); o2=opac.clone().detach().requires_grad_(True)
    def fwd_bwd():
        gids,radii,means2d,depths,conics = project(m2,q2,s2,o2,viewmat,K,W,H,sh_degree)
        if impl=="A":
            tpg,isect_ids,flat = isect_A(means2d,radii,depths,torch.zeros_like(gids),gids,W,H)
        elif impl=="B":
            tpg,isect_ids,flat = isect_BC(B,means2d,conics,torch.sigmoid(o2)[gids],depths,W,H,1)
        else:
            tpg,isect_ids,flat = isect_BC(C,means2d,conics,torch.sigmoid(o2)[gids],depths,W,H,1)
        offsets=isect_offset_encode(isect_ids,1,TW,TH)
        dirs=m2[gids].detach().clone()-cam_center.unsqueeze(0)
        col=torch.clamp_min(spherical_harmonics(sh_degree,dirs,shs[gids])+0.5,0.0).contiguous()
        orr=torch.sigmoid(o2[gids]).contiguous()
        r=raster(means2d,conics,col,orr,W,H,offsets,flat)
        loss=torch.mean((r[0]-target)**2)
        torch.cuda.synchronize(); t0=time.time()
        grads=torch.autograd.grad(loss,[m2,q2,s2,o2],allow_unused=True)
        torch.cuda.synchronize(); dt=time.time()-t0
        return grads,dt
    # warm + measure (take best backward time)
    best_bwd=1e9
    for _ in range(3):
        grads,dt=fwd_bwd(); best_bwd=min(best_bwd,dt)
    res["m3_backward_s"]=best_bwd
    res["m3_forward_plus_backward_s"]=t_fwd+best_bwd
    # --- M4 full training step incl optimizer ---
    if opt is not None:
        params=[m2,q2,s2,o2]
        def train_step():
            opt.zero_grad(set_to_none=True)
            gids,radii,means2d,depths,conics = project(m2,q2,s2,o2,viewmat,K,W,H,sh_degree)
            if impl=="A":
                tpg,isect_ids,flat = isect_A(means2d,radii,depths,torch.zeros_like(gids),gids,W,H)
            elif impl=="B":
                tpg,isect_ids,flat = isect_BC(B,means2d,conics,torch.sigmoid(o2)[gids],depths,W,H,1)
            else:
                tpg,isect_ids,flat = isect_BC(C,means2d,conics,torch.sigmoid(o2)[gids],depths,W,H,1)
            offsets=isect_offset_encode(isect_ids,1,TW,TH)
            dirs=m2[gids].detach().clone()-cam_center.unsqueeze(0)
            col=torch.clamp_min(spherical_harmonics(sh_degree,dirs,shs[gids])+0.5,0.0).contiguous()
            orr=torch.sigmoid(o2[gids]).contiguous()
            r=raster(means2d,conics,col,orr,W,H,offsets,flat)
            loss=torch.mean((r[0]-target)**2)
            loss.backward()
            torch.cuda.synchronize(); t0=time.time()
            opt.step()
            torch.cuda.synchronize(); dt=time.time()-t0
            return dt
        for _ in range(2): train_step()  # warm
        best_step=1e9
        for _ in range(5): best_step=min(best_step,train_step())
        res["m4_optimizer_step_s"]=best_step
        res["m4_full_training_step_s"]=t_fwd+best_bwd+best_step
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenes",default="room,garden,bicycle")
    ap.add_argument("--ncam",type=int,default=8)
    ap.add_argument("--impls",default="A,B,C")
    ap.add_argument("--res",default="native,down2")
    ap.add_argument("--train",type=int,default=1)  # 1=do M4 (needs optimizer)
    ap.add_argument("--outdir",default="/root/pbr0/out")
    ap.add_argument("--smoke",type=int,default=0)
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    scenes=args.scenes.split(","); impls=args.impls.split(","); reses=args.res.split(",")
    out={"config":dict(scenes=scenes,ncam=args.ncam,impls=impls,res=reses,tile=TILE),"scenes":{}}
    for scene in scenes:
        sc=load_scene(scene)
        ids=sc["ids"]; n=min(args.ncam,len(ids))
        cam_idx=[int(ids[int(i*len(ids)/n)]) for i in range(n)] if args.smoke==0 else [int(ids[0])]
        means,quats,scales,opac,shs=to_gpu(sc)
        out["scenes"][scene]=dict(N=sc["N"],SH=sc["SH"],W=sc["W"],H=sc["H"],cameras={})
        print("====",scene,"N=",sc["N"],"cams=",cam_idx,flush=True)
        # single shared optimizer over the 4 param groups (for M4 timing)
        opt=torch.optim.Adam([{"params":[means]},{"params":[quats]},{"params":[scales]},{"params":[opac]}],lr=1e-3)
        for ci in cam_idx:
            gp=gt_path(scene,sc["imgs"][ci]["name"])
            entry={}
            for rk in reses:
                if rk=="down2": W,H=sc["W"]//2,sc["H"]//2
                else: W,H=sc["W"],sc["H"]
                target=load_gt_tensor(gp,W,H) if gp else torch.rand(H,W,3,device=dev)
                entry[rk]={}
                for impl in impls:
                    r=measure_camera(sc,ci,W,H,impl,means,quats,scales,opac,shs,sc["SH"],target,opt if args.train else None)
                    entry[rk][impl]=r
                    print("  ",scene,"cam",ci,rk,impl,"m2=%.4f m1=%.4f nnz=%d"%(r["m2_forward_pipeline_s"],r["m1_isect_s"],r["isect_count_nnz"]),flush=True)
            out["scenes"][scene]["cameras"][str(ci)]=entry
    with open(os.path.join(args.outdir,"PBR0_BENCHMARK.json"),"w") as f:
        json.dump(out,f,indent=1)
    print("BENCH_DONE",flush=True)

if __name__=="__main__":
    main()
