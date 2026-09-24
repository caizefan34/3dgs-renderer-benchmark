#!/usr/bin/env python3
"""NEW-6: Backward Execution Metadata Reuse Screening.

Can forward-generated metadata (last_ids, per-tile termination depth,
active-pixel count) be used as a SCHEDULER for backward execution,
not just as data?

Hypothesis: last_ids + tile occupancy predict backward tail waste,
enabling per-tile backward execution policy.
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
W, H, TILE = 1920, 1080, 16; TW, TH = math.ceil(W/TILE), math.ceil(H/TILE); DEV="cuda"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--camera",type=int,default=5); p.add_argument("--scene",default="room")
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

    torch.set_grad_enabled(False)
    rgb,a,meta=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),width=W,height=H,
        near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,packed=False,tile_size=TILE,
        backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
    
    # Get last_ids via raw forward kernel (gsplat 1.5.3 doesn't expose via meta)
    from gsplat.cuda._wrapper import _make_lazy_cuda_func
    _, iid, fid = gsplat.isect_tiles(meta["means2d"].contiguous(), meta["radii"].contiguous(),
        meta["depths"].contiguous(), TILE, TW, TH, sort=True)
    roff = gsplat.isect_offset_encode(iid, 1, TW, TH).contiguous()
    n_sorted = int(fid.numel())
    dirs = (cam.camera_center.to(DEV) - means)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    colors_fwd = gsplat.spherical_harmonics(3, dirs, shs).unsqueeze(0)
    rc_out, ra_out, last_tup = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")(
        meta["means2d"].contiguous(), meta["conics"].contiguous(),
        colors_fwd.contiguous(), meta["opacities"].contiguous(), bg, None,
        W, H, TILE, roff.contiguous(), fid.contiguous())
    last_ids_raw = last_tup[0]
    if last_ids_raw.dim() == 3:
        last_ids_raw = last_ids_raw[0]
    torch.cuda.synchronize()
    
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=True)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    n_sorted=int(fid.numel()); starts=roff[0].reshape(-1).long().tolist(); ends=starts[1:]+[n_sorted]
    
    ends_list = ends
    tile_metadata=[]
    for ti in range(len(starts)):
        lo=starts[ti]
        hi=ends_list[ti] if ti<len(ends_list) else n_sorted
        n=hi-lo
        if n<=0:
            tile_metadata.append({"tile":ti,"n_sorted":0,"prediction":"inactive","tail_waste":0})
            continue
        ty=ti//TW; tx=ti%TW
        
        # Get per-pixel last_ids for this tile
        tile_last_ids=[]
        for wy in range(TILE):
            for wx in range(TILE):
                py=ty*TILE+wy; px=tx*TILE+wx
                if py<H and px<W:
                    lid=last_ids_raw[py,px].item()
                    # lid is index in sorted array, convert to relative
                    lid_rel=lid-lo
                    tile_last_ids.append(max(lid_rel,0))
        
        max_last=max(tile_last_ids) if tile_last_ids else 0
        min_last=min(tile_last_ids) if tile_last_ids else 0
        avg_last=np.mean(tile_last_ids) if tile_last_ids else 0
        
        # Waste estimation: sorted positions > max_last are all wasted
        waste=n-max(0,max_last)-1
        waste_ratio=waste/max(n,1) if n>0 else 0
        
        # What fraction of pixels have terminated before depth x?
        depths=np.array(tile_last_ids)/max(n,1)
        active_pixels=sum(1 for d in depths if d>0.5)  # pixels where last_id > 50% depth
        early_term=sum(1 for d in depths if d<0.25)  # pixels where last_id < 25% depth
        
        # Can we predict waste from tile workload stats?
        # Hypothetical predictor: tail_waste_ratio = f(max_last, n, avg_last)
        tile_metadata.append({
            "tile":ti,"n_sorted":n,"max_last":max_last,"min_last":min_last,
            "avg_last":round(float(avg_last),1),
            "waste_positions":waste,"waste_ratio":round(float(waste_ratio),4),
            "active_pixels_beyond_50pct":active_pixels,
            "early_term_pixels":early_term,
        })
    
    # Aggregate: what share of T_iter does the tail waste represent?
    total_waste=sum(tm["waste_positions"] for tm in tile_metadata)
    total_sorted=sum(tm["n_sorted"] for tm in tile_metadata)
    overall_waste_ratio=total_waste/max(total_sorted,1)
    
    # Predictive analysis: which tile stats best predict waste_ratio?
    waste_ratios=np.array([tm["waste_ratio"] for tm in tile_metadata if tm["n_sorted"]>0])
    n_sorted_vals=np.array([tm["n_sorted"] for tm in tile_metadata if tm["n_sorted"]>0])
    corr_waste_nsorted=float(np.corrcoef(waste_ratios,n_sorted_vals)[0,1]) if len(waste_ratios)>2 else 0
    
    # Are the tiles with highest waste_pct predictable?
    high_waste=[tm for tm in tile_metadata if tm["waste_ratio"]>0.5]
    n_high_waste=len(high_waste)
    
    out={
        "schema_version":2,"phase":"NEW-6 backward execution metadata reuse",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":{"scene":args.scene,"resolution":f"{W}x{H}","tile_size":TILE,"camera":args.camera},
        "total_sorted":total_sorted,"total_waste":total_waste,
        "overall_waste_ratio":round(float(overall_waste_ratio),4),
        "tiles_high_waste":n_high_waste,
        "waste_nsorted_correlation":round(corr_waste_nsorted,3),
        "sample_tiles":tile_metadata[:min(20,len(tile_metadata))],
        "analysis": (
            f"Overall backward sorted position waste: {overall_waste_ratio*100:.1f}% (tail >last_ids). "
            f"Correlation waste-n_sorted: {corr_waste_nsorted:.3f}. "
            f"{n_high_waste} tiles have >50% waste."
        ),
        "verdict":"KEEP" if overall_waste_ratio>0.1 else "DROP",
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")
    print(f"Total waste: {total_waste}/{total_sorted} ({overall_waste_ratio*100:.1f}%)")
    print(f"High-waste tiles (>50%): {n_high_waste}")

if __name__=="__main__": main()
