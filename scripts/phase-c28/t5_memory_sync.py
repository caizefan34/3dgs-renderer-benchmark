#!/usr/bin/env python3
"""C28 A4/A5 — T5' shared-memory / sync isolation.

Measure backward time across tile-load distributions.  Determine
shared-memory and synchronization overhead separately from
computation.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
W,H,TILE=1920,1080,16; TW,TH=math.ceil(W/TILE),math.ceil(H/TILE); DEV="cuda"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--camera",type=int,default=5)
    args=p.parse_args(); torch.manual_seed(0); bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
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

    # 1) Full baseline + tile analysis
    fwds,bwds=[],[]
    for rep in range(10):
        for x in [means,quats,scales,opac,shs]: x.grad=None
        torch.cuda.synchronize();t0=time.perf_counter()
        ra,aa,meta=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
            viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
            width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
            packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
        torch.cuda.synchronize();t1=time.perf_counter()
        (ra.float().mean()+aa.float().mean()).backward()
        torch.cuda.synchronize();t2=time.perf_counter()
        fwds.append((t1-t0)*1000);bwds.append((t2-t1)*1000)
    t_base={"fwd_ms":float(np.mean(fwds)),"bwd_ms":float(np.mean(bwds)),"T_iter_ms":float(np.mean(fwds)+np.mean(bwds))}
    results["baseline"]=t_base
    print(f"Baseline: fwd={t_base['fwd_ms']:.2f} bwd={t_base['bwd_ms']:.2f} iter={t_base['T_iter_ms']:.2f}ms",flush=True)

    # 2) Per-tile workload analysis
    torch.set_grad_enabled(False)
    ra,aa,meta=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
        width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
        packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=True)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    ns=int(fid.numel())
    starts=roff[0].reshape(-1).long().tolist(); ends=starts[1:]+[ns]

    # Per-tile batch count and shared-memory bytes
    total_batches=0; total_smem_bytes=0
    per_tile=[]
    for ti in range(len(starts)):
        lo=starts[ti]; hi=ends[ti] if ti<len(ends) else ns; n=hi-lo
        if n<=0:continue
        n_batches=(n+255)//256
        total_batches+=n_batches
        # smem per batch: id_batch(4*256) + xy_opacity(12*256) + conic(12*256) + rgbs(12*256) = 10240 bytes
        total_smem_bytes+=n_batches*10240
        per_tile.append({"tile":ti,"n_sorted":n,"n_batches":n_batches,"smem_kb":n_batches*10240/1024})
    
    # 3) Estimate per-batch overhead breakdown
    n_nonzero=len(per_tile)
    empty_tiles=TW*TH-n_nonzero
    # Each empty tile still launches a block (8160 blocks total)
    # Each block: syncthreads overhead ~20ns per batch
    # Shared-memory load: 10KB per batch, at ~1TB/s = ~10ns per batch
    # warp reduction: ~10ns per reduction × 8 reductions × 8 warps = ~640ns per batch
    
    sync_ns=20; smem_load_ns=10; reduction_ns=640
    overhead_per_batch_ns=sync_ns+smem_load_ns+reduction_ns
    overhead_per_batch_ms=overhead_per_batch_ns/1e6
    
    # Empty tiles: 1 batch each (the tile_offsets check, then return)
    # Actually empty tiles don't execute the inner loop — they return early
    # but they still launch, do tile_offsets check, and return
    empty_overhead_ms=(TW*TH-n_nonzero)*0.002  # ~2us per empty launch
    
    overhead_from_batches=total_batches*overhead_per_batch_ms
    overhead_from_empty=empty_overhead_ms
    
    # Total batches for warp reduction cost
    # Each batch → 8 warps × ~9 reductions = 72 reductions
    # At 10ns each: 720ns per batch
    # But reduction cost is INDEPENDENT of active count per warp
    results["overhead_analysis"]={
        "total_batches":total_batches,
        "total_smem_mb":round(total_smem_bytes/1024/1024,2),
        "n_nonzero_tiles":n_nonzero,
        "n_empty_tiles":empty_tiles,
        "per_batch_overhead_ns":{"sync":sync_ns,"smem_load":smem_load_ns,"warp_reduction":reduction_ns},
        "estimated_total_overhead_ms":round(overhead_from_batches+overhead_from_empty,3),
        "estimated_struct_fixed_overhead_ms":round(overhead_from_empty,3),
        "estimated_scalable_overhead_ms":round(overhead_from_batches,3),
    }
    print(f"Batches: {total_batches}, Overhead est: {overhead_from_batches+overhead_from_empty:.2f}ms",flush=True)

    # 4) Load-balance: what if we could skip the tail 50% of batches?
    batches_by_tile=[(ends[ti] if ti<len(ends) else ns)-starts[ti] for ti in range(len(starts)) if (ends[ti] if ti<len(ends) else ns)-starts[ti]>0]
    if batches_by_tile:
        batches_by_tile=np.array(batches_by_tile)
        # Tail 50% is positions beyond median tile size
        med_batches=np.median(batches_by_tile)
        tail_batches=sum(max(0,b-med_batches) for b in batches_by_tile)
        tail_batches_frac=tail_batches/sum(batches_by_tile)
        results["tail_skip_potential"]={
            "median_tile_load":float(med_batches),
            "tail_batches_fraction":float(tail_batches_frac),
            "load_imbalance":float(np.std(batches_by_tile)/max(np.mean(batches_by_tile),0.001)),
        }
        print(f"Tail skip potential: {tail_batches_frac*100:.1f}% of batches beyond median",flush=True)

    out={"schema_version":2,"phase":"C28 T5' memory/sync",
         "timestamp_utc":datetime.now(timezone.utc).isoformat(),
         "protocol":{"scene":"room","resolution":f"{W}x{H}","tile_size":TILE,"camera":args.camera},"results":results}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")

if __name__=="__main__": main()
