#!/usr/bin/env python3
"""GPU7 — H Gaussian Lifecycle Backup Research.

Investigate whether Gaussian lifecycle stages (new/old, large/small,
high/low opacity) create measurable cost differences that could be
exploited by a lifecycle-aware execution policy.

Key question: Cost(Gaussian) vs Contribution(Gaussian) — are there
Gaussians that consume disproportionate cost for minimal contribution?
"""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

W,H,TILE=1920,1080,16; TW,TH=math.ceil(W/TILE),math.ceil(H/TILE); DEV="cuda"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--camera",type=int,default=5)
    args=p.parse_args()
    torch.manual_seed(0)
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    cam=cams[args.camera]; bg=torch.zeros(1,3,device=DEV)
    means=scene["xyz"].detach().clone(); scales=scene["scales"].detach().clone().exp()
    opac=scene["opacity"].detach().clone(); quats=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1)
    shs=scene["shs"].detach().clone()

    # Get forward to determine which Gaussians contribute
    torch.set_grad_enabled(False)
    rgb,a,meta=gsplat.rasterization(
        means=means.requires_grad_(True),quats=quats.requires_grad_(True),scales=scales.requires_grad_(True),
        opacities=opac.requires_grad_(True),colors=shs.requires_grad_(True),
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
        width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,
        sh_degree=3,packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",
        sparse_grad=False,absgrad=False,rasterize_mode="classic")
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),
        meta["depths"].contiguous(),TILE,TW,TH,sort=True)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    n_sorted=int(fid.numel())

    # Build per-Gaussian cost proxy: number of sorted positions this G occupies
    g_cost=np.zeros(means.shape[0],dtype=np.float32)
    starts=roff[0].reshape(-1).long().tolist()
    ends=starts[1:]+[n_sorted]
    for ti,(lo,hi) in enumerate(zip(starts,ends)):
        n=hi-lo
        if n<=0: continue
        gids=fid[lo:hi].cpu().numpy()
        for g in set(gids):
            if g<len(g_cost):
                g_cost[g]+=1

    # Contribution proxy: opacity (how much this G affects rendering)
    g_opac=opac.cpu().numpy().flatten()
    # Size proxy: scale magnitude
    g_scales=scales.cpu().numpy()
    g_size=np.linalg.norm(g_scales if g_scales.ndim==2 else g_scales.reshape(-1,3),axis=1)
    
    # Visible vs invisible
    visible_mask=g_cost>0
    n_visible=int(visible_mask.sum())
    n_invisible=int((~visible_mask).sum())

    # Cost percentile analysis among visible Gaussians
    vis_cost=g_cost[visible_mask]
    vis_opac=g_opac[visible_mask]
    vis_size=g_size[visible_mask]

    # Top-10% most expensive Gaussians: what share of cost?
    sorted_idx=np.argsort(vis_cost)[::-1]
    top10pct=max(len(vis_cost)//10,1)
    top10_cost_share=vis_cost[sorted_idx[:top10pct]].sum()/max(vis_cost.sum(),1)

    # Bottom-50% cheapest: what share?
    bottom50pct=sorted_idx[-max(len(vis_cost)//2,1):]
    bottom50_cost_share=vis_cost[bottom50pct].sum()/max(vis_cost.sum(),1)

    # Cost vs opacity correlation
    corr=np.corrcoef(vis_cost,vis_opac)[0,1] if len(vis_cost)>1 else 0
    cost_size_corr=np.corrcoef(vis_cost,vis_size)[0,1] if len(vis_cost)>1 else 0

    # Do small Gaussians always cost less?
    # Gaussian size vs intersection count
    tiny=vis_size<0.01  # very small screen-space footprint
    if tiny.sum()>0:
        tiny_cost_share=vis_cost[tiny].sum()/max(vis_cost.sum(),1)
        tiny_g_share=tiny.sum()/len(vis_cost)
    else:
        tiny_cost_share=tiny_g_share=0

    out={
        "schema_version":2,"phase":"H Gaussian lifecycle backup",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":{"scene":"room","resolution":f"{W}x{H}","tile_size":TILE,"camera":args.camera},
        "scene_stats":{"total_gaussians":int(means.shape[0]),"visible":int(n_visible),"invisible":int(n_invisible)},
        "cost_analysis":{
            "top_10pct_cost_share":float(top10_cost_share),
            "bottom_50pct_cost_share":float(bottom50_cost_share),
            "cost_opacity_correlation":float(corr),
            "cost_size_correlation":float(cost_size_corr),
            "tiny_gaussian_share":float(tiny_g_share),
            "tiny_gaussian_cost_share":float(tiny_cost_share),
            "visible_cost_range": [float(vis_cost.min()), float(np.percentile(vis_cost,50)), float(vis_cost.max())],
        },
        "analysis":(
            f"Top 10% of visible Gaussians account for {top10_cost_share*100:.1f}% of intersection cost. "
            f"Bottom 50% account for {bottom50_cost_share*100:.1f}%. "
            f"Cost-opacity correlation: {corr:.3f}. "
            f"Cost-size correlation: {cost_size_corr:.3f}. "
            f"Tiny Gaussians (<0.01 size): {tiny_g_share*100:.1f}% of visible, "
            f"costing {tiny_cost_share*100:.1f}% of total."
        ),
        "verdict":(
            "KEEP" if top10_cost_share>0.3
            else "DROP"
        ),
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")
    print(f"  Vis: {n_visible}  Inv: {n_invisible}")
    print(f"  Top10% cost share: {top10_cost_share*100:.1f}%  Bottom50%: {bottom50_cost_share*100:.1f}%")

if __name__=="__main__": main()
