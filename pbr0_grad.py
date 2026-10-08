#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PBR0 gradient audit + preregistered nondeterminism noise floor.
Repeat identical baseline (A) >=Na times, identical TFC1 (B) >=Nb times.
Compute A-A / B-B / A-B cosine + relL2, global AND per-parameter.
Freeze the A-A noise envelope BEFORE evaluating A-B. Preserve legacy cosine>=0.99999 gate.
EPIC-05 A100 only. Measurement only; no algorithm changes."""
import sys, os, json, math, argparse, time
sys.path.insert(0, "/root/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/root/tfc2/kernel")
sys.path.insert(0, "/root/tfc0/kernel")
import numpy as np, torch
import tfc1_rowwise as B
import tfc0_lib as L
from gsplat import (fully_fused_projection, rasterization, isect_offset_encode,
                    rasterize_to_pixels, spherical_harmonics, isect_tiles)
from benchmark_framework.scene import load_ply
from PIL import Image

dev="cuda"; TILE=L.TILE; EPS2D=L.EPS2D; NEAR=L.NEAR; FAR=L.FAR; RC=0.0
REPO="/root/3dgs-renderer-benchmark"
def colmap_dir(s): return os.path.join(REPO,"data","datasets","mipnerf360",s,"sparse","0")
def ply_official(s): return os.path.join(REPO,"data","official","mipnerf360",s,"point_cloud.ply")
def ply_train(s): return "/root/tfc2/train/%s/point_cloud/iteration_30000/point_cloud.ply" % s
def gt_path(scene,name):
    base=os.path.join(REPO,"data","datasets","mipnerf360",scene,"images")
    if os.path.splitext(name)[1].lower() in ('.jpg','.png','.jpeg'):
        p=os.path.join(base,name)
        if os.path.exists(p): return p
    for ext in ('.JPG','.jpg','.PNG','.png','.jpeg'):
        p=os.path.join(base,name+ext)
        if os.path.exists(p): return p
    return None
def load_gt_tensor(path,W,H):
    img=Image.open(path).convert("RGB")
    if img.size[0]!=W or img.size[1]!=H: img=img.resize((W,H),Image.BILINEAR)
    return torch.from_numpy(np.asarray(img,dtype=np.float32)/255.0).to(dev)
def arr(v):
    v=v.numpy() if hasattr(v,"numpy") else np.asarray(v); return v.astype(np.float32)
def load_scene(scene,ply):
    cd=colmap_dir(scene) if scene in ("room","garden","bicycle") else None
    data=load_ply(ply,device="cpu")
    xyz=arr(data["xyz"]); scales=np.exp(arr(data["scales"])); quats=arr(data["rotations"])
    opac=arr(data["opacity"]); shs=arr(data["shs"]); SH=int(round(math.sqrt(shs.shape[1])-1))
    out=dict(scene=scene,N=int(xyz.shape[0]),SH=SH,W=data and None,xyz=xyz,scales=scales,quats=quats,opac=opac,shs=shs)
    if cd:
        imgs=L.read_images_bin(os.path.join(cd,"images.bin")); cams=L.read_cameras_bin(os.path.join(cd,"cameras.bin"))
        out.update(W=cams["W"],H=cams["H"],ids=sorted(imgs.keys()),imgs=imgs,cams=cams)
    else:
        # train: read colmap for cam151 image name only
        tcd="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/train/sparse/0"
        imgs=L.read_images_bin(os.path.join(tcd,"images.bin"))
        out.update(W=None,H=None,ids=sorted(imgs.keys()),imgs=imgs)
    return out
def to_gpu(sc):
    return (torch.from_numpy(sc["xyz"]).to(dev),torch.from_numpy(sc["quats"]).to(dev),
            torch.from_numpy(sc["scales"]).to(dev),torch.from_numpy(sc["opac"]).to(dev),
            torch.from_numpy(sc["shs"]).to(dev))
def cam_tensors(sc,ci,W,H):
    imgs,cams=sc["imgs"],sc["cams"] if "cams" in sc else (sc["imgs"],None)
    if cams is not None:
        fx,fy,cx,cy=cams["params"][:4]; fx=fx*W/sc["W"]; fy=fy*H/sc["H"]; cx=cx*W/sc["W"]; cy=cy*H/sc["H"]
    else:
        fx,fy,cx,cy=W*0.5,H*0.5,W*0.5,H*0.5  # fallback (train cam151 uses its own K below)
    R=L.qvec2rot(imgs[ci]["qvec"]); t=np.array(imgs[ci]["tvec"])
    viewmat=np.eye(4); viewmat[:3,:3]=R; viewmat[:3,3]=t
    c2w=np.linalg.inv(viewmat); cam_center=c2w[:3,3]
    viewmat_t=torch.tensor(viewmat,dtype=torch.float32,device=dev).unsqueeze(0)
    K_t=torch.tensor(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32),device=dev).unsqueeze(0)
    cam_center_t=torch.tensor(cam_center,dtype=torch.float32,device=dev)
    return viewmat_t,K_t,cam_center_t,imgs[ci]["name"]

# train cam151 needs real intrinsics; load train cams properly
def load_train_scene():
    sc=load_scene("train",ply_train("train"))
    tcd="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/train/sparse/0"
    cams=L.read_cameras_bin(os.path.join(tcd,"cameras.bin"))
    sc["cams"]=cams; sc["W"]=cams["W"]; sc["H"]=cams["H"]
    return sc

def grad_A(means,quats,scales,opac,shs,viewmat,K,W,H,sh_degree,target):
    m2=means.clone().detach().requires_grad_(True); q2=quats.clone().detach().requires_grad_(True)
    s2=scales.clone().detach().requires_grad_(True); o2=opac.clone().detach().requires_grad_(True)
    r,_,_=rasterization(means=m2,covars=None,quats=q2,scales=s2,opacities=torch.sigmoid(o2),colors=shs,
        viewmats=viewmat,Ks=K,width=W,height=H,eps2d=EPS2D,near_plane=NEAR,far_plane=FAR,radius_clip=RC,
        packed=True,render_mode="RGB",sh_degree=sh_degree)
    loss=torch.mean((r[0]-target)**2)
    g=torch.autograd.grad(loss,[m2,q2,s2,o2],allow_unused=True)
    return {"means":g[0],"quats":g[1],"scales":g[2],"opac":g[3]}

def grad_B(means,quats,scales,opac,shs,viewmat,K,W,H,sh_degree,target,cam_center):
    m2=means.clone().detach().requires_grad_(True); q2=quats.clone().detach().requires_grad_(True)
    s2=scales.clone().detach().requires_grad_(True); o2=opac.clone().detach().requires_grad_(True)
    bid,cid,gids,radii,means2d,depths,conics,comp=fully_fused_projection(means=m2,covars=None,quats=q2,scales=s2,
        viewmats=viewmat,Ks=K,width=W,height=H,eps2d=EPS2D,near_plane=NEAR,far_plane=FAR,radius_clip=RC,
        packed=True,opacities=torch.sigmoid(o2),calc_compensations=False)
    TW=(W+TILE-1)//TILE; TH=(H+TILE-1)//TILE
    tpg,isect_ids,flat=B.footprint_intersect(means2d.detach(),conics.detach(),torch.sigmoid(o2)[gids].detach(),
        depths.detach(),None,float(TILE),W,H,RC,1)
    offsets=isect_offset_encode(isect_ids,1,TW,TH)
    dirs=m2[gids].detach().clone()-cam_center.unsqueeze(0)
    col=torch.clamp_min(spherical_harmonics(sh_degree,dirs,shs[gids])+0.5,0.0).contiguous()
    orr=torch.sigmoid(o2[gids]).contiguous()
    r,_=rasterize_to_pixels(means2d,conics,col,orr,W,H,TILE,offsets,flat,packed=True)
    loss=torch.mean((r[0]-target)**2)
    g=torch.autograd.grad(loss,[m2,q2,s2,o2],allow_unused=True)
    return {"means":g[0],"quats":g[1],"scales":g[2],"opac":g[3]}

def flat_cat(gd):
    parts=[g.reshape(-1) for g in [gd["means"],gd["quats"],gd["scales"],gd["opac"]] if g is not None]
    return torch.cat(parts)
def cosine(a,b):
    a=a.flatten(); b=b.flatten()
    return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-30))
def relL2(a,b):
    a=a.flatten(); b=b.flatten()
    return float((a-b).norm()/(a.norm()+1e-30))

def pairwise_stats(grads):
    # grads: list of grad dicts. returns global + per-param pairwise cosine/relL2 lists
    n=len(grads); gc=[]; gr=[]; per={k:{"cos":[],"rel":[]} for k in ["means","quats","scales","opac"]}
    for i in range(n):
        for j in range(i+1,n):
            fi=flat_cat(grads[i]); fj=flat_cat(grads[j])
            gc.append(cosine(fi,fj)); gr.append(relL2(fi,fj))
            for k in per:
                a=grads[i][k]; b=grads[j][k]
                if a is not None and b is not None:
                    per[k]["cos"].append(cosine(a,b)); per[k]["rel"].append(relL2(a,b))
    return dict(global_cos=gc,global_rel=gr,per=per)

def ab_compare(ga,gb):
    # paired by index: A[i] vs B[i]
    cos=[]; rel=[]; per={k:{"cos":[],"rel":[]} for k in ["means","quats","scales","opac"]}
    n=min(len(ga),len(gb))
    for i in range(n):
        fi=flat_cat(ga[i]); fj=flat_cat(gb[i]); cos.append(cosine(fi,fj)); rel.append(relL2(fi,fj))
        for k in per:
            a=ga[i][k]; b=gb[i][k]
            if a is not None and b is not None:
                per[k]["cos"].append(cosine(a,b)); per[k]["rel"].append(relL2(a,b))
    return dict(global_cos=cos,global_rel=rel,per=per,n_pairs=n)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--na",type=int,default=12)
    ap.add_argument("--nb",type=int,default=12)
    ap.add_argument("--cameras",default="room:1,39,78,117,156,195,234,273;garden:1,24,47,70,93,116,139,162;bicycle:1,25,49,73,98,122,146,170")
    ap.add_argument("--train_cam151","1") if False else ap.add_argument("--train_cam151",type=int,default=1)
    ap.add_argument("--outdir",default="/root/pbr0/out")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    scene_cams={}
    for blk in args.cameras.split(";"):
        s,cs=blk.split(":"); scene_cams[s]=[int(x) for x in cs.split(",")]
    results={}
    # ---- scene cameras ----
    for scene,cams in scene_cams.items():
        sc=load_scene(scene,ply_official(scene)); means,quats,scales,opac,shs=to_gpu(sc)
        results[scene]={}
        for ci in cams:
            W,H=sc["W"],sc["H"]
            gp=gt_path(scene,sc["imgs"][ci]["name"]); target=load_gt_tensor(gp,W,H) if gp else torch.rand(H,W,3,device=dev)
            viewmat,K,cam_center,name=cam_tensors(sc,ci,W,H)
            t0=time.time()
            gA=[grad_A(means,quats,scales,opac,shs,viewmat,K,W,H,sc["SH"],target) for _ in range(args.na)]
            gB=[grad_B(means,quats,scales,opac,shs,viewmat,K,W,H,sc["SH"],target,cam_center) for _ in range(args.nb)]
            torch.cuda.synchronize()
            # STEP 1: freeze A-A envelope BEFORE evaluating A-B
            aa=pairwise_stats(gA)
            aa_env=dict(
                cos_floor_pct5=float(np.percentile(aa["global_cos"],5)) if aa["global_cos"] else None,
                cos_median=float(np.median(aa["global_cos"])) if aa["global_cos"] else None,
                cos_min=float(np.min(aa["global_cos"])) if aa["global_cos"] else None,
                rel_ceiling_pct95=float(np.percentile(aa["global_rel"],95)) if aa["global_rel"] else None,
                rel_median=float(np.median(aa["global_rel"])) if aa["global_rel"] else None,
                rel_max=float(np.max(aa["global_rel"])) if aa["global_rel"] else None,
                n_pairs=len(aa["global_cos"]))
            # STEP 2: B-B
            bb=pairwise_stats(gB)
            # STEP 3: A-B (envelope already frozen)
            ab=ab_compare(gA,gB)
            ab_cos_med=float(np.median(ab["global_cos"])) if ab["global_cos"] else None
            ab_rel_med=float(np.median(ab["global_rel"])) if ab["global_rel"] else None
            real_diff = (ab_cos_med is not None and aa_env["cos_floor_pct5"] is not None and ab_cos_med < aa_env["cos_floor_pct5"]) or \
                        (ab_rel_med is not None and aa_env["rel_ceiling_pct95"] is not None and ab_rel_med > aa_env["rel_ceiling_pct95"])
            results[scene][str(ci)]=dict(
                camera_id=ci, name=name, W=W, H=H,
                nA=args.na, nB=args.nb,
                aa_envelope=aa_env,
                bb=dict(global_cos_median=float(np.median(bb["global_cos"])) if bb["global_cos"] else None,
                        global_rel_median=float(np.median(bb["global_rel"])) if bb["global_rel"] else None,
                        cos_min=float(np.min(bb["global_cos"])) if bb["global_cos"] else None),
                ab=dict(global_cos_median=ab_cos_med, global_rel_median=ab_rel_med,
                        cos_min=float(np.min(ab["global_cos"])) if ab["global_cos"] else None,
                        n_pairs=ab["n_pairs"],
                        per={k:dict(cos_median=float(np.median(v["cos"])) if v["cos"] else None,
                                        rel_median=float(np.median(v["rel"])) if v["rel"] else None) for k,v in ab["per"].items()}),
                ab_vs_aa_envelope="REAL_DIFF" if real_diff else "WITHIN_NOISE",
                **{"legacy_gate_cos_ge_0.99999": bool(ab_cos_med is not None and ab_cos_med>=0.99999)},
                per_aa={k:dict(cos_median=float(np.median(v["cos"])) if v["cos"] else None) for k,v in aa["per"].items()},
                per_bb={k:dict(cos_median=float(np.median(v["cos"])) if v["cos"] else None) for k,v in bb["per"].items()},
                secs=round(time.time()-t0,2))
            print("  ",scene,"cam",ci,"A-Acos_med=%.6f A-Bcos_med=%.6f env_floor=%.6f -> %s"%(
                aa_env["cos_median"], ab_cos_med if ab_cos_med else -1, aa_env["cos_floor_pct5"] if aa_env["cos_floor_pct5"] else -1,
                results[scene][str(ci)]["ab_vs_aa_envelope"]),flush=True)
            # persist incrementally (fail-safe)
            with open(os.path.join(args.outdir,"PBR0_GRADIENT_AUDIT.json"),"w") as f:
                json.dump(dict(config=dict(na=args.na,nb=args.nb,scenes=scene_cams,train_cam151=bool(args.train_cam151)),
                               legacy_gate="cosine>=0.99999 (PRESERVED)",results=results),f,indent=1)
    # ---- train cam151 diagnostic ----
    if args.train_cam151:
        sc=load_train_scene(); means,quats,scales,opac,shs=to_gpu(sc)
        ci=151; W,H=sc["W"],sc["H"]
        gp=gt_path("train",sc["imgs"][ci]["name"]); target=load_gt_tensor(gp,W,H) if gp else torch.rand(H,W,3,device=dev)
        viewmat,K,cam_center,name=cam_tensors(sc,ci,W,H)
        t0=time.time()
        gA=[grad_A(means,quats,scales,opac,shs,viewmat,K,W,H,sc["SH"],target) for _ in range(args.na)]
        gB=[grad_B(means,quats,scales,opac,shs,viewmat,K,W,H,sc["SH"],target,cam_center) for _ in range(args.nb)]
        torch.cuda.synchronize()
        aa=pairwise_stats(gA); bb=pairwise_stats(gB); ab=ab_compare(gA,gB)
        aa_env=dict(cos_floor_pct5=float(np.percentile(aa["global_cos"],5)) if aa["global_cos"] else None,
                    cos_median=float(np.median(aa["global_cos"])) if aa["global_cos"] else None,
                    cos_min=float(np.min(aa["global_cos"])) if aa["global_cos"] else None,
                    rel_ceiling_pct95=float(np.percentile(aa["global_rel"],95)) if aa["global_rel"] else None,
                    rel_median=float(np.median(aa["global_rel"])) if aa["global_rel"] else None,
                    n_pairs=len(aa["global_cos"]))
        ab_cos_med=float(np.median(ab["global_cos"])) if ab["global_cos"] else None
        ab_rel_med=float(np.median(ab["global_rel"])) if ab["global_rel"] else None
        real_diff=(ab_cos_med is not None and aa_env["cos_floor_pct5"] is not None and ab_cos_med<aa_env["cos_floor_pct5"]) or \
                  (ab_rel_med is not None and aa_env["rel_ceiling_pct95"] is not None and ab_rel_med>aa_env["rel_ceiling_pct95"])
        results.setdefault("train",{})
        results["train"]["151"]=dict(camera_id=151,name=name,W=W,H=H,nA=args.na,nB=args.nb,
            aa_envelope=aa_env,
            bb=dict(global_cos_median=float(np.median(bb["global_cos"])) if bb["global_cos"] else None,
                    global_rel_median=float(np.median(bb["global_rel"])) if bb["global_rel"] else None),
            ab=dict(global_cos_median=ab_cos_med,global_rel_median=ab_rel_med,n_pairs=ab["n_pairs"],
                    per={k:dict(cos_median=float(np.median(v["cos"])) if v["cos"] else None,
                                    rel_median=float(np.median(v["rel"])) if v["rel"] else None) for k,v in ab["per"].items()}),
            ab_vs_aa_envelope="REAL_DIFF" if real_diff else "WITHIN_NOISE",
            **{"legacy_gate_cos_ge_0.99999": bool(ab_cos_med is not None and ab_cos_med>=0.99999)},
            per_aa={k:dict(cos_median=float(np.median(v["cos"])) if v["cos"] else None) for k,v in aa["per"].items()},
            per_bb={k:dict(cos_median=float(np.median(v["cos"])) if v["cos"] else None) for k,v in bb["per"].items()},
            secs=round(time.time()-t0,2))
        print("   train cam151 A-Acos_med=%.6f A-Bcos_med=%.6f -> %s"%(
            aa_env["cos_median"], ab_cos_med if ab_cos_med else -1, results["train"]["151"]["ab_vs_aa_envelope"]),flush=True)
        with open(os.path.join(args.outdir,"PBR0_GRADIENT_AUDIT.json"),"w") as f:
            json.dump(dict(config=dict(na=args.na,nb=args.nb,scenes=scene_cams,train_cam151=True),
                           legacy_gate="cosine>=0.99999 (PRESERVED)",results=results),f,indent=1)
    print("GRAD_DONE",flush=True)

if __name__=="__main__":
    main()
