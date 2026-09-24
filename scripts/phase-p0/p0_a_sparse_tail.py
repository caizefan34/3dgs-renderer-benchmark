#!/usr/bin/env python3
"""P0-A: T5' Sparse-Tail Specialization Prototype (v3 — simple & robust)."""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
W,H,TILE=1920,1080,16; TW,TH=math.ceil(W/TILE),math.ceil(H/TILE); DEV="cuda"
from gsplat.cuda._wrapper import _make_lazy_cuda_func

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--camera",type=int,default=5); p.add_argument("--scene",default="room"); p.add_argument("--reps",type=int,default=10)
    args=p.parse_args(); torch.manual_seed(0); bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    scene=load_ply(str(ROOT/f"data/official/mipnerf360/{args.scene}/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/f"data/official/mipnerf360/{args.scene}/cameras.json"),device=DEV),W,H)
    cam=cams[args.camera]
    means=scene["xyz"].detach().clone().requires_grad_(True)
    quats=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1).requires_grad_(True)
    scales=scene["scales"].detach().clone().exp().requires_grad_(True)
    opac=scene["opacity"].detach().clone().requires_grad_(True)
    shs=scene["shs"].detach().clone().requires_grad_(True)
    
    # Warmup
    for _ in range(3):
        for x in [means,quats,scales,opac,shs]: x.grad=None
        ra,aa,_=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
        (ra.float().mean()+aa.float().mean()).backward()
    torch.cuda.synchronize()

    results={}

    # 1) Baseline timing
    fwds,bwds=[],[]
    for rep in range(args.reps):
        for x in [means,quats,scales,opac,shs]: x.grad=None
        torch.cuda.synchronize(); t0=time.perf_counter()
        rgb,a,_=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
        torch.cuda.synchronize(); t1=time.perf_counter()
        (rgb.float().mean()+a.float().mean()).backward()
        torch.cuda.synchronize(); t2=time.perf_counter()
        fwds.append((t1-t0)*1000); bwds.append((t2-t1)*1000)
    t_base={"fwd_ms":float(np.mean(fwds)),"bwd_ms":float(np.mean(bwds)),"T_iter_ms":float(np.mean(fwds)+np.mean(bwds))}
    results["baseline"]=t_base
    print(f"Baseline: fwd={t_base['fwd_ms']:.2f} bwd={t_base['bwd_ms']:.2f} iter={t_base['T_iter_ms']:.2f}ms",flush=True)

    # 2) Depth tail efficiency (separate forward with no autograd needed)
    torch.set_grad_enabled(False)
    means_d=means.detach(); quats_d=quats.detach(); scales_d=scales.detach(); opac_d=opac.detach(); shs_d=shs.detach()
    rgb,a,meta=gsplat.rasterization(means=means_d,quats=quats_d,scales=scales_d,opacities=opac_d,colors=shs_d,
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
        width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
        packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
    
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=True)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    ns=int(fid.numel())
    
    # Raw forward for last_ids
    dirs=(cam.camera_center.to(DEV)-means_d); dirs=dirs/dirs.norm(dim=-1,keepdim=True)
    cfwd=gsplat.spherical_harmonics(3,dirs,shs_d).unsqueeze(0)
    _,_,lt=_make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")(
        meta["means2d"].contiguous(),meta["conics"].contiguous(),cfwd.contiguous(),
        meta["opacities"].contiguous(),bg,None,W,H,TILE,roff.contiguous(),fid.contiguous())
    lid=lt[0]
    if lid.dim()==3: lid=lid[0]
    torch.cuda.synchronize()
    torch.set_grad_enabled(True)
    
    starts=roff[0].reshape(-1).long().tolist()
    ends=starts[1:]+[ns]
    intervals=[(0,0.5),(0.5,0.75),(0.75,0.90),(0.90,0.95),(0.95,0.99),(0.99,1.0)]
    ilabels=["0-50%","50-75%","75-90%","90-95%","95-99%","99-100%"]
    
    isort={l:0 for l in ilabels}
    for ti in range(len(starts)):
        lo=starts[ti];hi=ends[ti] if ti<len(ends) else ns;n=hi-lo
        if n<=0:continue
        for si in range(lo,hi):
            dp=(si-lo)/max(n,1)
            for idx,(dlo,dhi) in enumerate(intervals):
                if dlo<=dp<dhi: isort[ilabels[idx]]+=1;break
    
    total_sorted=sum(isort.values())
    tail_sorted=sum(isort[l] for l in ["75-90%","90-95%","95-99%","99-100%"])
    results["tail"]={
        "total_sorted":total_sorted,"tail_75_100_sorted":tail_sorted,
        "tail_75_100_fraction":round(tail_sorted/max(total_sorted,1),4),
        "interval_counts":isort
    }
    print(f"Tail 75-100%: {tail_sorted}/{total_sorted} ({tail_sorted/max(total_sorted,1)*100:.1f}%)",flush=True)

    # 3) Active-lane sweep (fresh tensors each time)
    n_total=means.shape[0]
    fracs=[1.0,0.5,0.25,0.125,0.0625,0.03125,0.01,0.001]
    sweep=[]
    for sf in fracs:
        n=max(int(n_total*sf),1)
        m2=means[:n].detach().clone().requires_grad_(True)
        q2=quats[:n].detach().clone().requires_grad_(True)
        s2=scales[:n].detach().clone().requires_grad_(True)
        o2=opac[:n].detach().clone().requires_grad_(True)
        sh2=shs[:n].detach().clone().requires_grad_(True)
        fw,bw=[],[]
        for rep in range(args.reps):
            for x in [m2,q2,s2,o2,sh2]: x.grad=None
            torch.cuda.synchronize();t0=time.perf_counter()
            ra,aa,_=gsplat.rasterization(means=m2,quats=q2,scales=s2,opacities=o2,colors=sh2,
                viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
                width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
                packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
            torch.cuda.synchronize();t1=time.perf_counter()
            (ra.float().mean()+aa.float().mean()).backward()
            torch.cuda.synchronize();t2=time.perf_counter()
            fw.append((t1-t0)*1000);bw.append((t2-t1)*1000)
        sweep.append({"fraction":sf,"n":n,"bwd_ms":float(np.mean(bw)),"bwd_rel":float(np.mean(bw)/max(t_base['bwd_ms'],0.001))})
        print(f"  frac={sf:.4f} n={n:7d} bwd={np.mean(bw):.2f}ms rel={np.mean(bw)/max(t_base['bwd_ms'],0.001):.2f}",flush=True)
    results["sweep"]=sweep
    
    floor=sweep[-1]["bwd_ms"]
    results["feasibility"]={
        "structural_floor_ms":round(floor,2),"full_bwd_ms":round(t_base["bwd_ms"],2),
        "avoidable_pct":round((t_base["bwd_ms"]-floor)/max(t_base["bwd_ms"],0.001)*100,1),
    }
    print(f"\nStructural floor: {floor:.2f}ms ({floor/max(t_base['bwd_ms'],0.001)*100:.0f}% of full)",flush=True)
    
    out={"schema_version":2,"phase":"P0-A T5'","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":{"scene":args.scene,"resolution":f"{W}x{H}","tile_size":TILE,"camera":args.camera},"results":results}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}",flush=True)

if __name__=="__main__": main()
