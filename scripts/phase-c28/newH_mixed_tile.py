#!/usr/bin/env python3
"""C28 — NEW-H: Mixed execution by tile workload.

Classify tiles by load.  Can high-load and low-load tiles use different
execution strategies within the same iteration?
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
W,H,TILE=1920,1080,16;TW,TH=math.ceil(W/TILE),math.ceil(H/TILE);DEV="cuda"

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--camera",type=int,default=5)
    args=p.parse_args();torch.manual_seed(0);bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    cam=cams[args.camera]
    means=scene["xyz"].detach().clone().requires_grad_(True)
    quats=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1).requires_grad_(True)
    scales=scene["scales"].detach().clone().exp().requires_grad_(True)
    opac=scene["opacity"].detach().clone().requires_grad_(True)
    shs=scene["shs"].detach().clone().requires_grad_(True)
    
    torch.set_grad_enabled(False)
    ra,aa,meta=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
        width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
        packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=True)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    ns=int(fid.numel());starts=roff[0].reshape(-1).long().tolist();ends=starts[1:]+[ns]
    
    tile_loads=[(ends[ti] if ti<len(ends) else ns)-starts[ti] for ti in range(len(starts))]
    tile_loads=np.array(tile_loads)
    
    # Classify
    active=tile_loads[tile_loads>0]
    if len(active)==0:
        out={"verdict":"DROP","phase":"C28 NEW-H mixed tile execution"}
        Path(args.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(args.out,"w"),indent=2)
        return
    
    p25,p50,p75,p90,p99=np.percentile(active,[25,50,75,90,99])
    
    light=active[active<=p50];medium=active[(active>p50)&(active<=p90)];heavy=active[active>p90]
    
    out={"schema_version":2,"phase":"C28 NEW-H mixed tile execution",
         "timestamp_utc":datetime.now(timezone.utc).isoformat(),
         "protocol":{"scene":"room","resolution":f"{W}x{H}","tile_size":TILE,"camera":args.camera},
         "n_total_tiles":len(tile_loads),"n_active_tiles":int(len(active)),
         "distribution":{"p25":float(p25),"p50":float(p50),"p75":float(p75),"p90":float(p90),"p99":float(p99)},
         "tile_classes":{"light_count":int(len(light)),"light_mean":float(light.mean()),
             "medium_count":int(len(medium)),"medium_mean":float(medium.mean()),
             "heavy_count":int(len(heavy)),"heavy_mean":float(heavy.mean()),
             "heavy_fraction_of_total_sorted":float(heavy.sum()/active.sum())},
         "analysis":(
             f"Active tiles: {len(active)}. "
             f"Heavy tiles (P90+): {len(heavy)} tiles, {heavy.sum()/max(active.sum(),1)*100:.0f}% of total sorted positions. "
             + ("High workload concentration — heavy-tile specialization could cover most cost."
                if heavy.sum()/max(active.sum(),1)>0.4
                else "Workload is distributed — mixed execution offers limited benefit.")
         ),
         "verdict":"KEEP" if heavy.sum()/max(active.sum(),1)>0.4 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")
    print(f"Heavy tiles: {len(heavy)} ({heavy.sum()/max(active.sum(),1)*100:.0f}% of sorted)")

if __name__=="__main__":main()
