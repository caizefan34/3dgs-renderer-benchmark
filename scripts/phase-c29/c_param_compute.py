#!/usr/bin/env python3
"""C29-C: SH compute scheduling overhead."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, gsplat
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import W, H, TILE, DEV

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True)
    args=p.parse_args()
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    cam=cams[0]; bg=torch.zeros(1,3,device=DEV)
    m=scene["xyz"].detach().clone().requires_grad_(True)
    q=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1).requires_grad_(True)
    s=scene["scales"].detach().clone().exp().requires_grad_(True)
    o=scene["opacity"].detach().clone().requires_grad_(True)
    sh=scene["shs"].detach().clone().requires_grad_(True)
    # SH3 fwd
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(20):
        for x in [m,q,s,o,sh]: x.grad=None
        ra,aa,_=gsplat.rasterization(means=m,quats=q,scales=s,opacities=o,colors=sh,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
        (ra.float().mean()+aa.float().mean()).backward()
    torch.cuda.synchronize(); sh_t=(time.perf_counter()-t0)/20*1000
    # SH0 fwd
    sh0=sh[:,:1,:].contiguous().requires_grad_(True)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(20):
        for x in [m,q,s,o,sh0]: x.grad=None
        ra,aa,_=gsplat.rasterization(means=m,quats=q,scales=s,opacities=o,colors=sh0,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=0,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
        (ra.float().mean()+aa.float().mean()).backward()
    torch.cuda.synchronize(); no_sh_t=(time.perf_counter()-t0)/20*1000
    out={"schema_version":2,"phase":"C29-C","sh3_ms":round(sh_t,3),"sh0_ms":round(no_sh_t,3),
        "sh3_overhead_pct":round((sh_t/no_sh_t-1)*100,1),
        "verdict":"KEEP" if sh_t>no_sh_t*1.5 else "MAYBE" if sh_t>no_sh_t*1.2 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out} SH3={sh_t:.3f} SH0={no_sh_t:.3f}")
if __name__=="__main__": main()
