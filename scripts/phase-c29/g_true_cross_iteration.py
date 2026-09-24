#!/usr/bin/env python3
"""C29-G: True cross-iteration workload persistence."""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, gsplat
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import GaussianModel, W, H, TILE, TW, TH, DEV

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--n_iters",type=int,default=150)
    args=p.parse_args(); bg=torch.zeros(1,3,device=DEV)
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    cam=cams[0]
    m=GaussianModel(scene,3,DEV); gt=m.render(cam).unsqueeze(0).permute(0,3,1,2).contiguous(); del m; torch.cuda.empty_cache()
    model=GaussianModel(scene,3,DEV); opt=model.get_optimizer()
    wl=[]
    for step in range(args.n_iters):
        with torch.no_grad():
            ra,aa,meta=gsplat.rasterization(means=model.means,quats=model.quats,scales=torch.exp(model.scales),
                opacities=torch.sigmoid(model.opacities),colors=model.shs,
                viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),
                width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,
                packed=False,tile_size=TILE,backgrounds=bg,render_mode="RGB",rasterize_mode="classic")
            _,iid,fid=gsplat.isect_tiles(meta["means2d"].contiguous(),meta["radii"].contiguous(),meta["depths"].contiguous(),TILE,TW,TH,sort=False)
            roff=gsplat.isect_offset_encode(iid,1,TW,TH).contiguous()
            ns=int(fid.numel()) if fid is not None else 0; ioff=roff[0].reshape(-1).cpu().numpy() if roff.numel()>0 else np.array([0])
            tl=np.diff(np.concatenate([ioff,[ns]])); nz=tl[tl>0]; n_nz=int(len(nz))
            p90=float(np.percentile(nz,90)) if len(nz)>0 else 0; p50=float(np.percentile(nz,50)) if len(nz)>0 else 0
        wl.append({"step":step,"n_gaussians":model.n,"n_isect":int(iid.numel()) if iid is not None else 0,
            "n_sorted":ns,"n_nz_tiles":n_nz,"p50":p50,"p90":p90})
        rc=model.render(cam).unsqueeze(0).permute(0,3,1,2); loss=(rc-gt).abs().mean()+0.2*(1.0-min(1.0,1.0))
        model.densify_and_prune(step,gt=0.0002,po=0.005)
        if model.n!=len(opt.param_groups[0]["params"][0]): opt=model.get_optimizer()
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad(): model.scales.clamp_(min=-10.,max=10.); model.opacities.clamp_(min=-10.,max=10.)
        if (step+1)%50==0: print(f"Step {step+1}: Gs={model.n} isect={wl[-1]['n_isect']}",flush=True)
    def lc(a):
        if len(a)<3 or np.std(a[:-1])==0 or np.std(a[1:])==0: return 0.
        return float(np.corrcoef(a[:-1],a[1:])[0,1])
    ni=np.array([w["n_isect"] for w in wl]); ng=np.array([w["n_gaussians"] for w in wl]); nn=np.array([w["n_nz_tiles"] for w in wl])
    out={"schema_version":2,"phase":"C29-G","corr_isect_lag1":lc(ni),"corr_gs_lag1":lc(ng),"corr_nz_lag1":lc(nn),
        "verdict":"KEEP" if abs(lc(ni))>0.6 else "MAYBE" if abs(lc(ni))>0.3 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out} corr={lc(ni):.3f}")
if __name__=="__main__": main()



