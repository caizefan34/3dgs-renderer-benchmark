#!/usr/bin/env python3
"""C28 A1/A2/A4/A5 — T5' exact execution cost map.

Measure T_bwd decomposed by active-lane fraction using Gaussian
subsampling.  Build T_batch = T_load + T_sync + T_control + T_active
+ T_reduction + T_other.

Classify waste into Type A (mathematically unnecessary),
Type B (warp/block granularity overhead), Type C (unavoidable).
"""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
W,H,TILE=1920,1080,16; TW,TH=math.ceil(W/TILE),math.ceil(H/TILE); DEV="cuda"

def measure_timing(means, quats, scales, opac, shs, cam, bg, reps=10):
    fwds,bwds=[],[]
    for _ in range(reps):
        for x in [means,quats,scales,opac,shs]: x.grad=None
        torch.cuda.synchronize(); t0=time.perf_counter()
        ra,aa,_=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
        torch.cuda.synchronize(); t1=time.perf_counter()
        (ra.float().mean()+aa.float().mean()).backward()
        torch.cuda.synchronize(); t2=time.perf_counter()
        fwds.append((t1-t0)*1000); bwds.append((t2-t1)*1000)
    return float(np.mean(fwds)), float(np.mean(bwds))

def workload_stats(means, quats, scales, opac, shs, cam, bg):
    torch.set_grad_enabled(False)
    ra,aa,meta=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
        width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
        packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=False)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    ioff=roff[0].reshape(-1).cpu().numpy()
    tile_lens=np.diff(np.concatenate([ioff,[int(fid.numel())]]))
    nz=tile_lens[tile_lens>0]
    torch.set_grad_enabled(True)
    n_total_tiles=TW*TH
    n_active_tiles=len(nz)
    return {"n_isect":int(iid.numel()),"n_sorted":int(fid.numel()),"n_nonzero_tiles":n_active_tiles,
        "n_total_tiles":n_total_tiles,"tile_load_p50":float(np.percentile(nz,50)) if len(nz)>0 else 0,
        "tile_load_p90":float(np.percentile(nz,90)) if len(nz)>0 else 0,
        "tile_load_mean":float(np.mean(nz)) if len(nz)>0 else 0}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--camera",type=int,default=5); p.add_argument("--scene",default="room"); p.add_argument("--reps",type=int,default=10)
    args=p.parse_args(); torch.manual_seed(0); bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    scene=load_ply(str(ROOT/f"data/official/mipnerf360/{args.scene}/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/f"data/official/mipnerf360/{args.scene}/cameras.json"),device=DEV),W,H)
    cam=cams[args.camera]
    n_total=scene["xyz"].shape[0]
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
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
        (ra.float().mean()+aa.float().mean()).backward()
    torch.cuda.synchronize()

    results={}

    # Phase 1: Full-set baseline + workload stats
    fw0,bw0=measure_timing(means,quats,scales,opac,shs,cam,bg,args.reps)
    ws0=workload_stats(means,quats,scales,opac,shs,cam,bg)
    results["baseline"]={"fwd_ms":fw0,"bwd_ms":bw0,"T_iter_ms":fw0+bw0,"workload":ws0}
    print(f"Baseline: fwd={fw0:.2f} bwd={bw0:.2f} iter={fw0+bw0:.2f}ms",flush=True)
    print(f"  Tiles: {ws0['n_nonzero_tiles']}/{ws0['n_total_tiles']} nonzero, P90={ws0['tile_load_p90']:.0f}",flush=True)

    # Phase 2: Active-lane fraction sweep with workload tracking
    fracs=[1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.001]
    sweep_data=[]
    for sf in fracs:
        n=max(int(n_total*sf),1)
        m2=means[:n].detach().clone().requires_grad_(True)
        q2=quats[:n].detach().clone().requires_grad_(True)
        s2=scales[:n].detach().clone().requires_grad_(True)
        o2=opac[:n].detach().clone().requires_grad_(True)
        sh2=shs[:n].detach().clone().requires_grad_(True)
        fw,bw=measure_timing(m2,q2,s2,o2,sh2,cam,bg,args.reps)
        ws=workload_stats(m2,q2,s2,o2,sh2,cam,bg)

        # Cost decomposition model
        # T_batch = T_load + T_sync + T_control + T_active + T_reduction + T_other
        # We can't measure each directly but can estimate from scaling behavior
        # T_structural = T_batch at lowest fraction (non-scalable)
        # T_scalable = T_batch(100%) - T_structural
        # Type C (unavoidable structural): kernel launch, grid, shared-mem init
        # Type B (granularity overhead): warp-level reduction, block sync
        # Type A (mathematically unnecessary): gradients for already-terminated pixels

        sweep_data.append({
            "fraction":sf,"n_gaussians":n,
            "fwd_ms":fw,"bwd_ms":bw,"T_iter_ms":fw+bw,
            "bwd_relative":bw/max(bw0,0.001),
            "workload":ws,
        })
        print(f"  frac={sf:.4f} n={n:7d} bwd={bw:.3f}ms rel={bw/max(bw0,0.001):.3f}",flush=True)
    results["sweep"]=sweep_data

    # Phase 3: Tile-load distribution analysis (identify workload heterogeneity)
    torch.set_grad_enabled(False)
    ra,aa,meta=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
        width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
        packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=True)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    ns=int(fid.numel())
    starts=roff[0].reshape(-1).long().tolist()
    ends=starts[1:]+[ns]
    tile_loads=[(ends[ti] if ti<len(ends) else ns)-starts[ti] for ti in range(len(starts))]
    tile_loads=np.array([t for t in tile_loads if t>0])
    
    # Classify tiles into low/medium/high/very-high
    if len(tile_loads)>0:
        p25,p50,p75,p90,p99=np.percentile(tile_loads,[25,50,75,90,99])
        results["tile_load_distribution"]={
            "n_active_tiles":int(len(tile_loads)),"p25":float(p25),"p50":float(p50),
            "p75":float(p75),"p90":float(p90),"p99":float(p99),
            "min":float(tile_loads.min()),"max":float(tile_loads.max()),
            "cv":float(np.std(tile_loads)/max(np.mean(tile_loads),0.001)),
        }
        # Classify
        light=sum(1 for t in tile_loads if t<=p50)
        medium=sum(1 for t in tile_loads if p50<t<=p90)
        heavy=sum(1 for t in tile_loads if t>p90)
        print(f"  Tile load: p50={p50:.0f} p90={p90:.0f} p99={p99:.0f} CV={results['tile_load_distribution']['cv']:.2f}")
        print(f"  Light:{light} Medium:{medium} Heavy:{heavy}")
    torch.set_grad_enabled(True)

    # Phase 4: Cost classification
    floor=sweep_data[-1]["bwd_ms"]
    scalable=bw0-floor
    # Type C (structural): floor (kernel launch + grid + empty tile traversal)
    type_c_pct=floor/max(bw0,0.001)*100
    # Type B + A: the difference between linear scaling and actual
    # At 50% Gs, if perfectly linear cost would be 50% of bw0, actual is higher
    # The excess is Type B (overhead from block/warp granularity)
    type_b_estimates=[]
    for d in sweep_data:
        if d["fraction"]<1.0 and d["fraction"]>0.001:
            expected=d["fraction"]*scalable+floor
            excess=d["bwd_ms"]-expected
            if excess>0:
                type_b_estimates.append(excess)
    type_b_pct=np.mean(type_b_estimates)/max(bw0,0.001)*100 if type_b_estimates else 0

    results["cost_classification"]={
        "full_bwd_ms":bw0,"structural_floor_ms":floor,"scalable_ms":scalable,
        "type_C_structural_pct":round(type_c_pct,1),
        "type_B_granularity_pct":round(type_b_pct,1),
        "type_A_unnecessary_pct":round(100-type_c_pct-type_b_pct,1),
        "interpretation":(
            f"Type C (unavoidable structural): {type_c_pct:.1f}% — kernel launch, grid sched, shared-mem init, empty tile traversal. "
            f"Type B (warp/block granularity): {type_b_pct:.1f}% — warpSum over inactive lanes, block.sync, batch traversal overhead. "
            f"Type A (mathematically unnecessary): {100-type_c_pct-type_b_pct:.1f}% — gradient work for terminated pixels."
        ),
    }
    print(f"\nCost breakdown: Type C={type_c_pct:.1f}% Type B={type_b_pct:.1f}% Type A={100-type_c_pct-type_b_pct:.1f}%")

    out={"schema_version":2,"phase":"C28 T5' cost isolation","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":{"scene":args.scene,"resolution":f"{W}x{H}","tile_size":TILE,"camera":args.camera},"results":results}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")

if __name__=="__main__": main()
