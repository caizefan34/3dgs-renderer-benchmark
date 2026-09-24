#!/usr/bin/env python3
"""C29-D: Densification frequency adaptation."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import GaussianModel, W, H, DEV

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--n_steps",type=int,default=300)
    args=p.parse_args()
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    cam=cams[0]
    m=GaussianModel(scene,3,DEV); gt=m.render(cam).unsqueeze(0).permute(0,3,1,2).contiguous(); del m; torch.cuda.empty_cache()
    model=GaussianModel(scene,3,DEV); opt=model.get_optimizer()
    ev=[]
    for step in range(args.n_steps):
        rc=model.render(cam).unsqueeze(0).permute(0,3,1,2)
        loss=(rc-gt).abs().mean()+0.2*(1.0-min(1.0,1.0)).backward()
        nb=model.n; res=model.densify_and_prune(step,gt=0.0002,po=0.005)
        if model.n!=len(opt.param_groups[0]["params"][0]): opt=model.get_optimizer()
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad(): model.scales.clamp_(min=-10.,max=10.); model.opacities.clamp_(min=-10.,max=10.)
        if res["densified"]>0: ev.append({"step":step,"loss":float(loss.item()),"gs_before":int(nb),"gs_after":int(model.n)})
        if (step+1)%100==0: print(f"Step {step+1}: Gs={model.n}",flush=True)
    out={"schema_version":2,"phase":"C29-D","n_events":len(ev),"events":ev,
        "verdict":"MAYBE" if len(ev)>0 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out} events={len(ev)}")
if __name__=="__main__": main()



