#!/usr/bin/env python3
"""C28 — NEW-F: Cross-iteration workload persistence.

Measure corr(W_t, W_{t+1}) for intersection count, active tiles,
tile load, tail sparsity, backward time across consecutive cameras.
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
W,H,TILE=1920,1080,16;TW,TH=math.ceil(W/TILE),math.ceil(H/TILE);DEV="cuda"

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--n_cameras",type=int,default=60)
    args=p.parse_args();torch.manual_seed(0);bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    n=min(len(cams),args.n_cameras)
    n_total=scene["xyz"].shape[0]
    means=scene["xyz"].detach().clone().requires_grad_(True)
    quats=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1).requires_grad_(True)
    scales=scene["scales"].detach().clone().exp().requires_grad_(True)
    opac=scene["opacity"].detach().clone().requires_grad_(True)
    shs=scene["shs"].detach().clone().requires_grad_(True)
    
    # Warmup
    for _ in range(3):
        for x in [means,quats,scales,opac,shs]:x.grad=None
        ra,aa,_=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
            viewmats=cams[0].world_view_transform[None].contiguous(),Ks=cams[0].K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
        (ra.float().mean()+aa.float().mean()).backward()
    torch.cuda.synchronize()
    
    # Measure workload for each camera (done once, stable for static G set)
    workloads=[]
    for ci in range(n):
        cam=cams[ci]
        torch.set_grad_enabled(False)
        ra,aa,meta=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
        _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=False)
        ns=int(fid.numel())
        roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
        ioff=roff[0].reshape(-1).cpu().numpy();tile_lens=np.diff(np.concatenate([ioff,[ns]]))
        nz=tile_lens[tile_lens>0]
        torch.set_grad_enabled(True)
        workloads.append({"camera":ci,"n_isect":int(iid.numel()),"n_sorted":ns,
            "n_nz_tiles":int(len(nz)),"p50":float(np.percentile(nz,50)) if len(nz)>0 else 0,
            "p90":float(np.percentile(nz,90)) if len(nz)>0 else 0})
    
    # Correlation: W_t vs W_{t+1} in camera sequence order
    n_isect=np.array([w["n_isect"] for w in workloads])
    n_nz=np.array([w["n_nz_tiles"] for w in workloads])
    p90=np.array([w["p90"] for w in workloads])
    
    corr_isect=np.corrcoef(n_isect[:-1],n_isect[1:])[0,1]
    corr_nz=np.corrcoef(n_nz[:-1],n_nz[1:])[0,1]
    corr_p90=np.corrcoef(p90[:-1],p90[1:])[0,1]
    
    # Autocorrelation at lag 1
    out={"schema_version":2,"phase":"C28 NEW-F cross-iteration workload persistence",
         "timestamp_utc":datetime.now(timezone.utc).isoformat(),
         "protocol":{"scene":"room","resolution":f"{W}x{H}","tile_size":TILE,"n_cameras":n},
         "correlations":{"intersection_count_lag1":round(float(corr_isect),3),
             "nz_tiles_lag1":round(float(corr_nz),3),"p90_load_lag1":round(float(corr_p90),3)},
         "analysis":(
             f"Corr(isect_t, isect[t+1]): {corr_isect:.3f}. "
             f"Corr(nz_tiles): {corr_nz:.3f}. "
             f"Corr(P90 load): {corr_p90:.3f}. "
             + ("Workload is highly predictable across iterations."
                if abs(corr_isect)>0.5
                else "Workload varies significantly between consecutive cameras.")
         ),
         "verdict":"KEEP" if abs(corr_isect)>0.5 and abs(corr_nz)>0.5 else "MAYBE" if abs(corr_isect)>0.3 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")
    print(f"Corr isect={corr_isect:.3f} nz={corr_nz:.3f} p90={corr_p90:.3f}")

if __name__=="__main__":main()
