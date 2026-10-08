#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PBR0 reconciliation: TFC1(B) vs TFC2(C) intersection-membership equality and
rendered-output (pixel) identity, plus tightening PSNR-cost vs production A.
EPIC-05 A100 only. Measurement only; no algorithm changes."""
import sys, os, json, math, argparse
sys.path.insert(0, "/root/pbr0"); sys.path.insert(0, "/root/tfc2/kernel"); sys.path.insert(0, "/root/tfc0/kernel")
import numpy as np, torch
import pbr0_bench as PB
import tfc1_rowwise as B, tfc2_ablation as C, tfc0_lib as L
from gsplat import isect_offset_encode, spherical_harmonics

dev=PB.dev; TILE=PB.TILE

def psnr(a, b, cap=60.0):
    mse=torch.mean((a-b)**2).item()
    if mse<=1e-12: return float("inf")
    return float(min(cap, 10.0*math.log10(1.0/mse)))

def maxabs(a,b): return float(torch.max(torch.abs(a-b)).item())

def membership_equal(ib_B, flat_B, ib_C, flat_C):
    # sorted set equality of intersection ids and per-gauss flat lists
    sB=torch.sort(ib_B).values; sC=torch.sort(ib_C).values
    id_eq=bool(torch.equal(sB,sC))
    fB,_=torch.sort(flat_B); fC,_=torch.sort(flat_C)
    flat_eq=bool(torch.equal(fB,fC))
    return id_eq, flat_eq, int(sB.numel()), int(sC.numel())

def render_iso(sc, ci, W, H, means, quats, scales, opac, shs, sh_degree, target):
    viewmat,K,cam_center,name=PB.cam_tensors(sc,ci,W,H)
    TW=(W+TILE-1)//TILE; TH=(H+TILE-1)//TILE
    gids,radii,means2d,depths,conics=PB.project(means,quats,scales,opac,viewmat,K,W,H,sh_degree)
    cid0=torch.zeros_like(gids)
    out={}
    # A production (radii AABB)
    tpgA,ibA,flatA=PB.isect_A(means2d,radii,depths,cid0,gids,W,H)
    # B TFC1 ellipse
    tpgB,ibB,flatB=PB.isect_BC(B,means2d,conics,torch.sigmoid(opac)[gids],depths,W,H,1)
    # C TFC2 ellipse
    tpgC,ibC,flatC=PB.isect_BC(C,means2d,conics,torch.sigmoid(opac)[gids],depths,W,H,1)
    offA=isect_offset_encode(ibA,1,TW,TH); offB=isect_offset_encode(ibB,1,TW,TH); offC=isect_offset_encode(ibC,1,TW,TH)
    dirs=means[gids].detach().clone()-cam_center.unsqueeze(0)
    col=torch.clamp_min(spherical_harmonics(sh_degree,dirs,shs[gids])+0.5,0.0).contiguous()
    orr=torch.sigmoid(opac[gids]).contiguous()
    rA=PB.raster(means2d,conics,col,orr,W,H,offA,flatA)[0]
    rB=PB.raster(means2d,conics,col,orr,W,H,offB,flatB)[0]
    rC=PB.raster(means2d,conics,col,orr,W,H,offC,flatC)[0]
    id_eq,flat_eq,nnzB,nnzC=membership_equal(ibB,flatB,ibC,flatC)
    out.update(nnz_A=int(ibA.numel()),nnz_B=nnzB,nnz_C=nnzC,
        bc_id_equal=id_eq,bc_flat_equal=flat_eq,
        bc_nnz_equal=bool(nnzB==nnzC),
        psnr_BC=psnr(rB,rC), maxabs_BC=maxabs(rB,rC),
        psnr_AB=psnr(rA,rB), maxabs_AB=maxabs(rA,rB),
        psnr_A_GT=psnr(rA,target), psnr_B_GT=psnr(rB,target), psnr_C_GT=psnr(rC,target),
        psnr_loss_A_to_B=float(psnr(rA,target)-psnr(rB,target)) if math.isfinite(psnr(rA,target)) and math.isfinite(psnr(rB,target)) else None,
        mean_A=float(rA.mean()), mean_B=float(rB.mean()))
    return name, out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenes",default="room,garden,bicycle")
    ap.add_argument("--ncam",type=int,default=8)
    ap.add_argument("--res",default="native")
    ap.add_argument("--outdir",default="/root/pbr0/out")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    reses=args.res.split(",")
    cfg=dict(scenes=args.scenes.split(","),ncam=args.ncam,res=reses,tile=TILE,
             impls_compared=["B=tfc1_rowwise","C=tfc2_ablation"],reference="A:stock_radii_AABB")
    out={"config":cfg,"scenes":{}}
    for scene in args.scenes.split(","):
        sc=PB.load_scene(scene); ids=sc["ids"]; n=min(args.ncam,len(ids))
        cam_idx=[int(ids[int(i*len(ids)/n)]) for i in range(n)]
        means,quats,scales,opac,shs=PB.to_gpu(sc)
        out["scenes"][scene]=dict(N=sc["N"],SH=sc["SH"],W=sc["W"],H=sc["H"],cameras={})
        print("====",scene,"N=",sc["N"],"cams=",cam_idx,flush=True)
        for ci in cam_idx:
            gp=PB.gt_path(scene,sc["imgs"][ci]["name"])
            entry={}
            for rk in reses:
                W,H=(sc["W"]//2,sc["H"]//2) if rk=="down2" else (sc["W"],sc["H"])
                target=PB.load_gt_tensor(gp,W,H) if gp else torch.rand(H,W,3,device=dev)
                name,o=render_iso(sc,ci,W,H,means,quats,scales,opac,shs,sc["SH"],target)
                entry[rk]=o
                print("  ",scene,"cam",ci,rk,"nnz A/B/C=%d/%d/%d B==C(id)=%s flat=%s psnrBC=%.4f psnrAB=%.4f loss=%.4f"%(
                    o["nnz_A"],o["nnz_B"],o["nnz_C"],o["bc_id_equal"],o["bc_flat_equal"],
                    (o["psnr_BC"] if math.isfinite(o["psnr_BC"]) else 999),
                    (o["psnr_AB"] if math.isfinite(o["psnr_AB"]) else 999),
                    (o["psnr_loss_A_to_B"] if o["psnr_loss_A_to_B"] is not None else -1)),flush=True)
            out["scenes"][scene]["cameras"][str(ci)]=dict(name=name,resolutions=entry)
    with open(os.path.join(args.outdir,"PBR0_RECONCILIATION.json"),"w") as f:
        json.dump(out,f,indent=1)
    print("RECON_DONE",flush=True)

if __name__=="__main__":
    main()
