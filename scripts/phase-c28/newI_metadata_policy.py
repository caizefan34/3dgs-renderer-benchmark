#!/usr/bin/env python3
"""C28 — NEW-I: Backward metadata predictive scheduling.

Can forward-derived metadata predict which backward execution strategy
will be best?  Test predictors: active fraction, last_ids statistics,
tile length, tail sparsity.
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
W,H,TILE=1920,1080,16;TW,TH=math.ceil(W/TILE),math.ceil(H/TILE);DEV="cuda"
from gsplat.cuda._wrapper import _make_lazy_cuda_func

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--camera",type=int,default=5)
    args=p.parse_args();torch.manual_seed(0);bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    cam=cams[args.camera]
    means=scene["xyz"].detach().clone();quats=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1)
    scales=scene["scales"].detach().clone().exp();opac=scene["opacity"].detach().clone();shs=scene["shs"].detach().clone()
    n_total=means.shape[0]
    
    # Get forward metadata
    torch.set_grad_enabled(False)
    ra,aa,meta=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,
        colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
        width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
        packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=True)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    ns=int(fid.numel())
    
    # Get last_ids for per-tile analysis
    dirs=(cam.camera_center.to(torch.device("cuda"))-means);dirs=dirs/dirs.norm(dim=-1,keepdim=True)
    cfwd=gsplat.spherical_harmonics(3,dirs,shs).unsqueeze(0)
    _,_,lt=_make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")(
        meta["means2d"].contiguous(),meta["conics"].contiguous(),cfwd.contiguous(),
        meta["opacities"].contiguous(),bg,None,W,H,TILE,roff.contiguous(),fid.contiguous())
    lid=lt[0];lid=lid[0] if lid.dim()==3 else lid;torch.cuda.synchronize()
    
    starts=roff[0].reshape(-1).long().tolist();ends=starts[1:]+[ns]
    
    # Per-tile predictors
    tile_data=[]
    for ti in range(len(starts)):
        lo=starts[ti];hi=ends[ti] if ti<len(ends) else ns;n=hi-lo
        if n<=0:continue
        ty=ti//TW;tx=ti%TW
        # last_ids for pixels in this tile
        tile_lids=[]
        for wy in range(TILE):
            for wx in range(TILE):
                py=ty*TILE+wy;px=tx*TILE+wx
                if py<H and px<W:
                    tile_lids.append(int(lid[py,px].item()))
        max_lid=max(tile_lids);min_lid=min(tile_lids);mean_lid=np.mean(tile_lids)
        # Predictors:
        # 1. active_fraction = pixels with lid > 0.5*n / total
        active_frac=sum(1 for l in tile_lids if l>n*0.5)/max(len(tile_lids),1)
        # 2. tail_sparsity = (n - max_lid) / n
        tail_sparsity=(n-max_lid)/max(n,1)
        # 3. tile work = n
        # 4. lid_range = max_lid - min_lid
        lid_range=max_lid-min_lid
        tile_data.append({
            "tile":ti,"n_sorted":n,"max_lid":max_lid,"min_lid":min_lid,"mean_lid":float(mean_lid),
            "active_frac":round(float(active_frac),4),"tail_sparsity":round(float(tail_sparsity),4),
            "lid_range":lid_range})
    
    # Overall statistics
    n_active=sum(1 for t in tile_data if t["n_sorted"]>0)
    n_high_sparsity=sum(1 for t in tile_data if t["tail_sparsity"]>0.5)
    high_sparsity_cost=sum(t["n_sorted"] for t in tile_data if t["tail_sparsity"]>0.5)
    total_cost=sum(t["n_sorted"] for t in tile_data)
    
    # Correlation: tile workload vs tail_sparsity
    ns_arr=np.array([t["n_sorted"] for t in tile_data])
    ts_arr=np.array([t["tail_sparsity"] for t in tile_data])
    af_arr=np.array([t["active_frac"] for t in tile_data])
    corr_ns_ts=float(np.corrcoef(ns_arr,ts_arr)[0,1]) if len(ns_arr)>2 else 0
    corr_ns_af=float(np.corrcoef(ns_arr,af_arr)[0,1]) if len(ns_arr)>2 else 0
    
    out={"schema_version":2,"phase":"C28 NEW-I metadata predictive scheduling",
         "timestamp_utc":datetime.now(timezone.utc).isoformat(),
         "protocol":{"scene":"room","resolution":f"{W}x{H}","tile_size":TILE,"camera":args.camera},
         "n_active_tiles":n_active,"n_high_sparsity_tiles":n_high_sparsity,
         "high_sparsity_cost_share":round(float(high_sparsity_cost/max(total_cost,1)),4),
         "correlations":{"n_sorted_vs_tail_sparsity":round(corr_ns_ts,3),
             "n_sorted_vs_active_frac":round(corr_ns_af,3)},
         "sample_tiles":tile_data[:20],
         "analysis":(
             f"{n_high_sparsity} of {n_active} tiles have >50% tail sparsity. "
             f"These account for {high_sparsity_cost/max(total_cost,1)*100:.1f}% of sorted cost. "
             f"Corr(n_sorted, tail_sparsity): {corr_ns_ts:.3f}. "
             + ("Metadata strongly predicts sparse-tail waste — per-tile backward scheduling feasible."
                if high_sparsity_cost/max(total_cost,1)>0.2
                else "Sparse-tile cost is not concentrated enough for per-tile adaptation.")
         ),
         "verdict":"KEEP" if high_sparsity_cost/max(total_cost,1)>0.2 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")
    print(f"High-sparsity tiles: {n_high_sparsity}/{n_active} ({high_sparsity_cost/max(total_cost,1)*100:.1f}% of sorted)")
    torch.set_grad_enabled(True)

if __name__=="__main__":main()
