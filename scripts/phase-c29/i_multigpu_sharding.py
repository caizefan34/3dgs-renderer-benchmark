#!/usr/bin/env python3
"""C29-I: Multi-GPU camera sharding analysis."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, gsplat
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import W, H, TILE, DEV

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--n_cameras",type=int,default=100)
    p.add_argument("--n_gpus",type=int,default=8)
    args=p.parse_args(); bg=torch.zeros(1,3,device=DEV)
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    n=min(len(cams),args.n_cameras)
    m=scene["xyz"].detach().clone().requires_grad_(True)
    q=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1).requires_grad_(True)
    s=scene["scales"].detach().clone().exp().requires_grad_(True); o=scene["opacity"].detach().clone().requires_grad_(True)
    sh=scene["shs"].detach().clone().requires_grad_(True)
    cts=[]
    for ci in range(n):
        cam=cams[ci]
        for _ in range(3):
            for x in [m,q,s,o,sh]: x.grad=None
            ra,aa,_=gsplat.rasterization(means=m,quats=q,scales=s,opacities=o,colors=sh,
                viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
                width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
                packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
            (ra.float().mean()+aa.float().mean()).backward()
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(10):
            for x in [m,q,s,o,sh]: x.grad=None
            ra,aa,_=gsplat.rasterization(means=m,quats=q,scales=s,opacities=o,colors=sh,
                viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
                width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
                packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
            (ra.float().mean()+aa.float().mean()).backward()
        torch.cuda.synchronize(); ti=(time.perf_counter()-t0)/10*1000; cts.append(ti)
        if ci%20==0: print(f"Cam {ci}/{n}: {ti:.2f}ms",flush=True)
    cts=np.array(cts); gl=[cts[gi::args.n_gpus].sum() for gi in range(args.n_gpus)]
    ir=max(gl)/max(min(gl),1)
    out={"schema_version":2,"phase":"C29-I","n_cameras":n,"mean_t_iter_ms":round(float(cts.mean()),3),
        "t_iter_cv":round(float(cts.std()/max(cts.mean(),1e-10)),4),
        "load_imbalance_ratio":round(float(ir),3),"ngpu":args.n_gpus,
        "verdict":"KEEP" if ir>1.5 else "MAYBE" if ir>1.2 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out} imbalance={ir:.3f}")
if __name__=="__main__": main()
