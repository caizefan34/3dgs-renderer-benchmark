#!/usr/bin/env python3
"""C29-F: Camera workload/importance utility."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, gsplat
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import GaussianModel, W, H, TILE, DEV
def _psnr(x,y): m=((x-y)**2).mean(); return float(torch.tensor(-10*torch.log10(m.clamp(min=1e-10))))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--n_cameras",type=int,default=20)
    args=p.parse_args()
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    n=min(len(cams),args.n_cameras); bg=torch.zeros(1,3,device=DEV)
    m=scene["xyz"].detach().clone().requires_grad_(True)
    q=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1).requires_grad_(True)
    s=scene["scales"].detach().clone().exp().requires_grad_(True); o=scene["opacity"].detach().clone().requires_grad_(True)
    sh=scene["shs"].detach().clone().requires_grad_(True)
    cam_u=[]
    for ci in range(n):
        cam=cams[ci]
        for _ in range(5):
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
        torch.cuda.synchronize(); ti=(time.perf_counter()-t0)/10*1000
        for x in [m,q,s,o,sh]: x.grad=None
        ra,aa,_=gsplat.rasterization(means=m,quats=q,scales=s,opacities=o,colors=sh,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
        (ra.float().mean()+aa.float().mean()).backward()
        p2=GaussianModel(scene,3,DEV)
        with torch.no_grad(): gt=p2.render(cam).detach().unsqueeze(0).permute(0,3,1,2).contiguous(); del p2
        with torch.no_grad(): psnr=_psnr(ra.clamp(0,1).permute(2,0,1).unsqueeze(0),gt)
        gn=float(m.grad.norm().item()) if m.grad is not None else 0.
        util=gn*max(0.1,40-psnr)/max(ti,1e-6)
        cam_u.append({"camera":ci,"t_iter_ms":round(ti,3),"psnr":round(psnr,3),"grad_norm":round(gn,3),"utility":round(util,6)})
        print(f"Cam {ci}: PSNR={psnr:.2f} Ti={ti:.2f}ms util={util:.4f}",flush=True)
    utils=np.array([c["utility"] for c in cam_u])
    gini=float(np.abs(utils[:,None]-utils[None,:]).sum()/(2*len(utils)*utils.sum())) if utils.sum()>0 else 0
    out={"schema_version":2,"phase":"C29-F","cameras":cam_u,"utility_gini":round(gini,4),
        "verdict":"KEEP" if gini>0.4 else "MAYBE" if gini>0.2 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out} gini={gini:.3f}")
if __name__=="__main__": main()

