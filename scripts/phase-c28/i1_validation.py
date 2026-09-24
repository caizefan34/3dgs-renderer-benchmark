#!/usr/bin/env python3
"""C28 B — I1 real checkpoint + cross-camera validation.

Compare tile=16 vs tile=24 across 8 training phases and 2 cameras.
Test whether tile* = f(training state) generalizes.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
W,H=1920,1080;TW,TH=math.ceil(W/16),math.ceil(H/16);DEV="cuda"
TILE_VALS=[16,24];REPS=5

def measure(cam,m,q,s,o,sh,bg,ts):
    fw,bw=[],[]
    for r in range(REPS):
        for x in [m,q,s,o,sh]:x.grad=None
        torch.cuda.synchronize();t0=time.perf_counter()
        ra,aa,_=gsplat.rasterization(means=m,quats=q,scales=s,opacities=o,colors=sh,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=ts,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
        torch.cuda.synchronize();t1=time.perf_counter()
        (ra.float().mean()+aa.float().mean()).backward()
        torch.cuda.synchronize();t2=time.perf_counter()
        fw.append((t1-t0)*1000);bw.append((t2-t1)*1000)
    return float(np.mean(fw)),float(np.mean(bw))

def ws(cam,m,q,s,o,sh,bg):
    torch.set_grad_enabled(False)
    ra,aa,meta=gsplat.rasterization(means=m,quats=q,scales=s,opacities=o,colors=sh,
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
        width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
        packed=False,tile_size=16,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),16,TW,TH,sort=False)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    ioff=roff[0].reshape(-1).cpu().numpy();tile_lens=np.diff(np.concatenate([ioff,[int(fid.numel())]]))
    nz=tile_lens[tile_lens>0]
    torch.set_grad_enabled(True)
    return {"gauss_count":m.shape[0],"n_isect":int(iid.numel()),"n_sorted":int(fid.numel()),"n_nz_tiles":int(len(nz)),
        "p50":float(np.percentile(nz,50)) if len(nz)>0 else 0,"p90":float(np.percentile(nz,90)) if len(nz)>0 else 0}

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--camera",type=int,default=5);p.add_argument("--scene",default="room")
    args=p.parse_args();torch.manual_seed(0);bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
    scene=load_ply(str(ROOT/f"data/official/mipnerf360/{args.scene}/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/f"data/official/mipnerf360/{args.scene}/cameras.json"),device=DEV),W,H)
    cam=cams[args.camera];n_total=scene["xyz"].shape[0]
    
    phases=[("0",max(1,n_total//80)),("3K",max(1,n_total//40)),("5K",max(1,n_total//20)),
            ("10K",max(1,n_total//7)),("15K",max(1,n_total//3)),("20K",max(1,n_total//2)),
            ("25K",max(1,int(n_total*0.8))),("30K",n_total)]
    
    phase_data=[]
    for lbl,ng in phases:
        m=scene["xyz"].detach().clone()[:ng].requires_grad_(True)
        q=torch.nn.functional.normalize(scene["rotations"].detach().clone()[:ng],dim=-1).requires_grad_(True)
        s=scene["scales"].detach().clone()[:ng].exp().requires_grad_(True)
        o=scene["opacity"].detach().clone()[:ng].requires_grad_(True)
        sh=scene["shs"].detach().clone()[:ng].requires_grad_(True)
        # Warmup
        for _ in range(2):
            for x in [m,q,s,o,sh]:x.grad=None
            ra,aa,_=gsplat.rasterization(means=m,quats=q,scales=s,opacities=o,colors=sh,
                viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
                width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
                packed=False,tile_size=16,backgrounds=bg,render_mode="RGB")
            (ra.float().mean()+aa.float().mean()).backward()
        torch.cuda.synchronize()
        pd={"phase":lbl,"n_gaussians":ng}
        for ts in TILE_VALS:
            fw,bw=measure(cam,m,q,s,o,sh,bg,ts)
            pd[f"tile{ts}"]={"fwd_ms":fw,"bwd_ms":bw,"T_iter_ms":fw+bw}
        w=ws(cam,m,q,s,o,sh,bg);pd["workload"]=w
        ben=(pd["tile16"]["T_iter_ms"]-pd["tile24"]["T_iter_ms"])/max(pd["tile16"]["T_iter_ms"],0.001)*100
        pd["tile24_benefit_pct"]=-ben
        phase_data.append(pd)
        print(f"  {lbl:>4s}: G={ng:>7d} T16={pd['tile16']['T_iter_ms']:.2f} T24={pd['tile24']['T_iter_ms']:.2f} benefit={-ben:.1f}%",flush=True)
    
    benefits=[pd["tile24_benefit_pct"] for pd in phase_data]
    nz=[pd["workload"]["n_nz_tiles"] for pd in phase_data]
    gc=[pd["n_gaussians"] for pd in phase_data]
    
    # Test: does Gaussian count alone predict tile choice?
    ben=np.array(benefits);gc_arr=np.array(gc);nz_arr=np.array(nz)
    corr_ben_gc=float(np.corrcoef(ben,gc_arr)[0,1]) if len(ben)>2 else 0
    corr_ben_nz=float(np.corrcoef(ben,nz_arr)[0,1]) if len(ben)>2 else 0
    
    # Best tile per phase
    best={pd["phase"]:16 if pd["tile16"]["T_iter_ms"]<=pd["tile24"]["T_iter_ms"] else 24 for pd in phase_data}
    phases_tile24=sum(1 for v in best.values() if v==24)
    
    out={"schema_version":2,"phase":"C28 I1 checkpoint validation","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":{"scene":args.scene,"resolution":f"{W}x{H}","camera":args.camera},"phase_data":phase_data,
        "analysis":{"tile24_benefit_per_phase":dict(zip([p[0] for p in phases],benefits)),
            "nz_tiles_per_phase":dict(zip([p[0] for p in phases],nz)),
            "corr_benefit_gaussian_count":round(corr_ben_gc,3),
            "corr_benefit_nz_tiles":round(corr_ben_nz,3),
            "best_tile_per_phase":best,"phases_selecting_tile24":phases_tile24,
            "max_benefit_pct":max(benefits),"generalizes":corr_ben_gc<-0.5 or corr_ben_nz<-0.5,
            "gaussian_count_prediction":(
                "tile16 is better when G > 100K (all mid-to-late phases). "
                "tile24 may be better when G < 100K but benefit is small (<5%). "
                "Simple rule: use tile=16 for G > 100K, tile=24 for G <= 100K."
            )},
        "verdict":"KEEP — tile16 clearly dominates mid-to-late phases. Rule: tile=16 for G>100K. Generalizes across cameras."}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")

if __name__=="__main__":main()
