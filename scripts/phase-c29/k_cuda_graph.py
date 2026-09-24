#!/usr/bin/env python3
"""C29-K: CUDA graph training region analysis."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, gsplat
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import GaussianModel, W, H, TILE, DEV

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--n_iters",type=int,default=50)
    args=p.parse_args(); bg=torch.zeros(1,3,device=DEV)
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    cam=cams[0]
    m=GaussianModel(scene,3,DEV); gt=m.render(cam).unsqueeze(0).permute(0,3,1,2).contiguous(); del m; torch.cuda.empty_cache()
    model=GaussianModel(scene,3,DEV); opt=model.get_optimizer()
    for _ in range(5):
        rc=model.render(cam).unsqueeze(0).permute(0,3,1,2); loss=(rc-gt).abs().mean()+0.2*(1.0-min(1.0,1.0))
        model.densify_and_prune(0,gt=0.0002,po=0.005)
        if model.n!=len(opt.param_groups[0]["params"][0]): opt=model.get_optimizer()
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad(): model.scales.clamp_(min=-10.,max=10.); model.opacities.clamp_(min=-10.,max=10.)
    phases=[]; n_top=0
    for step in range(args.n_iters):
        e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e2=torch.cuda.Event(enable_timing=True); e3=torch.cuda.Event(enable_timing=True)
        e4=torch.cuda.Event(enable_timing=True)
        e0.record(); rc=model.render(cam).unsqueeze(0).permute(0,3,1,2)
        loss=(rc-gt).abs().mean()+0.2*(1.0-min(1.0,1.0)); e1.record()
        e2.record()
        res=model.densify_and_prune(step,gt=0.0002,po=0.005)
        if model.n!=len(opt.param_groups[0]["params"][0]): opt=model.get_optimizer()
        e3.record()
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad(): model.scales.clamp_(min=-10.,max=10.); model.opacities.clamp_(min=-10.,max=10.)
        e4.record()
        torch.cuda.synchronize()
        fm=e0.elapsed_time(e1); bm=e1.elapsed_time(e2); tm=e2.elapsed_time(e3); om=e3.elapsed_time(e4)
        phases.append({"fwd":fm,"bwd":bm,"topo":tm,"opt":om})
        if res["densified"]>0 or res["pruned"]>0: n_top+=1
    fwd_pct=np.mean([p["fwd"] for p in phases])/sum([np.mean([p[k] for p in phases]) for k in ["fwd","bwd","topo","opt"]])*100
    bwd_pct=np.mean([p["bwd"] for p in phases])/sum([np.mean([p[k] for p in phases]) for k in ["fwd","bwd","topo","opt"]])*100
    out={"schema_version":2,"phase":"C29-K","fwd_pct":round(fwd_pct,1),"bwd_pct":round(bwd_pct,1),
        "topo_change_iter_pct":round(n_top/args.n_iters*100,1),
        "graph_eligible_pct":round(fwd_pct+bwd_pct,1),
        "verdict":"KEEP" if fwd_pct+bwd_pct>70 else "MAYBE" if fwd_pct+bwd_pct>50 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out}")
if __name__=="__main__": main()



