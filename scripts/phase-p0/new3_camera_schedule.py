#!/usr/bin/env python3
"""NEW-3: Camera/Workload Cost-Aware Training Schedule.

Measure camera-level T_iter variation. Does camera ordering create
avoidable wall-clock cost? Measure for multiple cameras, test predictability.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
W, H, TILE = 1920, 1080, 16; DEV="cuda"

def measure_camera(cam_idx, cams, means, quats, scales, opac, shs, bg, reps=3):
    cam=cams[cam_idx]
    fwds,bwds=[],[]
    for r in range(reps):
        for x in [means, quats, scales, opac, shs]: x.grad=None
        torch.cuda.synchronize(); t0=time.perf_counter()
        rgb,a,_=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,
            sh_degree=3,packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
        torch.cuda.synchronize(); t1=time.perf_counter()
        (rgb.float().mean()+a.float().mean()).backward()
        torch.cuda.synchronize(); t2=time.perf_counter()
        fwds.append((t1-t0)*1000); bwds.append((t2-t1)*1000)
    return float(np.mean(fwds)), float(np.mean(bwds))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--scene",default="room"); p.add_argument("--n_cameras",type=int,default=30)
    args=p.parse_args(); torch.manual_seed(0); bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    scene=load_ply(str(ROOT/f"data/official/mipnerf360/{args.scene}/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/f"data/official/mipnerf360/{args.scene}/cameras.json"),device=DEV),W,H)
    n_total=scene["xyz"].shape[0]
    n=min(len(cams),args.n_cameras)
    means=scene["xyz"].detach().clone()[:n_total].requires_grad_(True)
    quats=torch.nn.functional.normalize(scene["rotations"].detach().clone()[:n_total],dim=-1).requires_grad_(True)
    scales=scene["scales"].detach().clone()[:n_total].exp().requires_grad_(True)
    opac=scene["opacity"].detach().clone()[:n_total].requires_grad_(True)
    shs=scene["shs"].detach().clone()[:n_total].requires_grad_(True)
    
    # Warmup
    for _ in range(3):
        for x in [means,quats,scales,opac,shs]: x.grad=None
        ra,aa,_=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
            viewmats=cams[0].world_view_transform[None].contiguous(),Ks=cams[0].K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,
            sh_degree=3,packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
        (ra.float().mean()+aa.float().mean()).backward()
    torch.cuda.synchronize()

    cam_results=[]
    for ci in range(n):
        fw,bw=measure_camera(ci,cams,means,quats,scales,opac,shs,bg)
        cam_results.append({"camera_id":ci,"fwd_ms":fw,"bwd_ms":bw,"T_iter_ms":fw+bw})
        print(f"  Camera {ci:3d}: T_iter={fw+bw:.2f}ms (fwd={fw:.2f} bwd={bw:.2f})",flush=True)
    
    T_iters=np.array([r["T_iter_ms"] for r in cam_results])
    cam_results.sort(key=lambda x: x["T_iter_ms"])
    
    min_t=T_iters.min(); max_t=T_iters.max(); mean_t=T_iters.mean(); std_t=T_iters.std()
    cv=std_t/mean_t  # coefficient of variation
    
    # Is the ordering stable? Repeat measurement for 3 cameras at each end
    cheapest_ids=[r["camera_id"] for r in cam_results[:3]]
    priciest_ids=[r["camera_id"] for r in cam_results[-3:]]
    
    recheck_cameras=cheapest_ids+priciest_ids
    recheck={}
    for ci in recheck_cameras:
        fw,bw=measure_camera(ci,cams,means,quats,scales,opac,shs,bg)
        recheck[ci]=fw+bw
    
    stable=(abs(recheck[cheapest_ids[0]]-cam_results[0]["T_iter_ms"])<0.5 and 
            abs(recheck[priciest_ids[-1]]-cam_results[-1]["T_iter_ms"])<0.5)
    
    # Camera pair cost ratio
    worst_best_ratio=max_t/min_t
    
    out={
        "schema_version":2,"phase":"NEW-3 camera workload cost-aware schedule",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":{"scene":args.scene,"resolution":f"{W}x{H}","tile_size":TILE,"n_cameras":n},
        "camera_results":cam_results,
        "statistics":{
            "T_iter_min_ms":round(float(min_t),3),"T_iter_max_ms":round(float(max_t),3),
            "T_iter_mean_ms":round(float(mean_t),3),"T_iter_std_ms":round(float(std_t),3),
            "cv":round(float(cv),3),"worst_best_ratio":round(float(worst_best_ratio),3),
        },
        "stability_test":{
            "cheapest_cameras":cheapest_ids,"priciest_cameras":priciest_ids,
            "recheck_values":recheck,"stable":stable,
        },
        "analysis": (
            f"Camera cost: min={min_t:.2f}ms max={max_t:.2f}ms ratio={worst_best_ratio:.2f}x CV={cv:.3f}. "
            f"Camera ranking is {'stable' if stable else 'unstable'} across rechecks. "
            + ("Camera ordering creates significant variation." if worst_best_ratio>1.5 else "Camera cost is relatively uniform.")
        ),
        "verdict":"KEEP" if worst_best_ratio>1.5 and cv>0.1 and stable else "DROP",
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2)
    print(f"\nSaved {args.out}")
    print(f"Camera cost range: {min_t:.1f}-{max_t:.1f}ms ({worst_best_ratio:.2f}x) CV={cv:.3f}")

if __name__=="__main__": main()
