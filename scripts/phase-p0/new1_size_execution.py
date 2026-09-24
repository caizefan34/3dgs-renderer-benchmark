#!/usr/bin/env python3
"""NEW-1: Cost-Aware Gaussian-Size Execution Path Screening.

C27 found: top 10% visible Gaussians = 61.8% of intersection cost.
Strong cost-size correlation (0.68). Question: can large Gaussians use a
different execution path?

Measure: footprint distribution, intersection cost per G, tile-span dist.
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
    p = argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--camera",type=int,default=5); p.add_argument("--scene",default="room")
    args=p.parse_args(); torch.manual_seed(0); bg=torch.zeros(1,3,device=DEV)
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    scene=load_ply(str(ROOT/f"data/official/mipnerf360/{args.scene}/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/f"data/official/mipnerf360/{args.scene}/cameras.json"),device=DEV),W,H)
    cam=cams[args.camera]
    means=scene["xyz"].detach().clone(); quats=torch.nn.functional.normalize(scene["rotations"].detach().clone(),dim=-1)
    scales=scene["scales"].detach().clone().exp(); opac=scene["opacity"].detach().clone(); shs=scene["shs"].detach().clone()

    torch.set_grad_enabled(False)
    rgb,a,meta=gsplat.rasterization(means=means.requires_grad_(True),quats=quats.requires_grad_(True),scales=scales.requires_grad_(True),
        opacities=opac.requires_grad_(True),colors=shs.requires_grad_(True),
        viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),width=W,height=H,
        near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB")
    _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=False)
    roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
    n_sorted=int(fid.numel()); starts=roff[0].reshape(-1).long().tolist(); ends=starts[1:]+[n_sorted]

    # Per-Gaussian intersection cost
    g_cost=np.zeros(means.shape[0],dtype=np.int32)
    g_tile_span=np.zeros(means.shape[0],dtype=np.int32)  # how many unique tiles
    g_screen_footprint=np.zeros(means.shape[0],dtype=np.float32)  # screen-space scale
    g_cost_per_intersection=np.zeros(means.shape[0],dtype=np.float32)

    for ti,(lo,hi) in enumerate(zip(starts,ends)):
        n=hi-lo
        if n<=0: continue
        gids=fid[lo:hi].cpu().numpy()
        for g in set(gids):
            if g<len(g_cost): g_cost[g]+=1; g_tile_span[g]+=1

    # Screen-space footprint: from scales (camera space) -> convert to screen projection
    # Use scale magnitude as proxy
    scales_np=scales.cpu().numpy()
    g_footprint=np.linalg.norm(scales_np if scales_np.ndim==2 else scales_np.reshape(-1,3),axis=1)
    
    visible=g_cost>0
    vis_cost=g_cost[visible]; vis_fp=g_footprint[visible]

    # Sorting by footprint
    sort_idx=np.argsort(vis_fp)[::-1]
    
    # Partition Gaussians into size-based groups (by percentile)
    n_vis=len(vis_cost)
    pcts=[0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    groups={}
    for pi in range(len(pcts)-1):
        lo_p=pcts[pi]; hi_p=pcts[pi+1]
        lo_idx=int(lo_p*n_vis); hi_idx=int(hi_p*n_vis)
        if hi_idx<=lo_idx: continue
        g_idx=sort_idx[lo_idx:hi_idx]
        groups[f"p{int(lo_p*100)}-{int(hi_p*100)}"]={
            "count":int(len(g_idx)),
            "total_intersection_cost":int(vis_cost[g_idx].sum()),
            "cost_share":float(vis_cost[g_idx].sum()/max(vis_cost.sum(),1)),
            "mean_footprint":float(vis_fp[g_idx].mean()),
            "mean_cost":float(vis_cost[g_idx].mean()),
        }

    # Memory behavior: how many Gaussians span many tiles?
    n_multi_tile=(g_tile_span>1).sum()
    cost_multi_tile=g_cost[g_tile_span>1].sum() if n_multi_tile>0 else 0
    
    # Hypothesis: large-footprint Gs have high per-intersection cost?
    for g in range(len(g_cost)):
        if g_cost[g]>0:
            g_cost_per_intersection[g]=float(g_footprint[g]/g_cost[g])
    
    corr_fp_cost=float(np.corrcoef(vis_fp,vis_cost)[0,1]) if len(vis_cost)>1 else 0
    
    # Check: cost concentration in large Gs vs small Gs
    # Top 5% by footprint -> what share of cost?
    top5_idx=sort_idx[:max(int(0.05*n_vis),1)]
    top5_cost_share=float(vis_cost[top5_idx].sum()/max(vis_cost.sum(),1))
    top5_fp_share=float(vis_fp[top5_idx].sum()/max(vis_fp.sum(),1))

    out={
        "schema_version":2,"phase":"NEW-1 cost-aware Gaussian-size execution path",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":{"scene":args.scene,"resolution":f"{W}x{H}","tile_size":TILE,"camera":args.camera},
        "visible_gaussians":int(n_vis),
        "size_groups":groups,
        "correlation_footprint_cost":round(corr_fp_cost,4),
        "top5pct_by_footprint_cost_share":round(top5_cost_share,4),
        "top5pct_by_footprint_fp_share":round(top5_fp_share,4),
        "multi_tile_gaussians":int(n_multi_tile),
        "multi_tile_cost_share":float(cost_multi_tile/max(g_cost.sum(),1)),
        "analysis": (
            f"Top 5% by screen-space footprint account for {top5_cost_share*100:.1f}% of intersection cost. "
            f"Correlation foot-print/cost: {corr_fp_cost:.3f}. "
            f"Bottom 50% smallest visible Gs account for "
            f"{groups.get('p0-50',{}).get('cost_share',0)*100:.1f}% of cost."
        ),
        "verdict":"KEEP" if top5_cost_share>0.3 else "DROP",
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")
    print(f"Top5% footprint -> {top5_cost_share*100:.1f}% cost. Corr fp:cost: {corr_fp_cost:.3f}")

if __name__=="__main__": main()
