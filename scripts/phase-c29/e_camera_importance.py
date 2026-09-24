#!/usr/bin/env python3
"""C29-E: Camera importance scheduling."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import GaussianModel, W, H, DEV
def _psnr(x,y): m=((x-y)**2).mean(); return float(torch.tensor(-10*torch.log10(m.clamp(min=1e-10))))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--n_cameras",type=int,default=15)
    args=p.parse_args()
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    n=min(len(cams),args.n_cameras)
    refs=[]
    for ci in range(n):
        m=GaussianModel(scene,3,DEV)
        with torch.no_grad(): ref=m.render(cams[ci]).unsqueeze(0).permute(0,3,1,2).contiguous()
        refs.append(ref); del m; torch.cuda.empty_cache()
    stats=[]
    for ci in range(n):
        torch.cuda.empty_cache(); model=GaussianModel(scene,3,DEV); opt=model.get_optimizer(); gt=refs[ci]
        for step in range(20):
            rc=model.render(cams[ci]).unsqueeze(0).permute(0,3,1,2)
            loss=(rc-gt).abs().mean()+0.2*(1.0-min(1.0,1.0)); loss.backward()
            model.densify_and_prune(step,gt=0.0002,po=0.005)
            if model.n!=len(opt.param_groups[0]["params"][0]): opt=model.get_optimizer()
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad(): model.scales.clamp_(min=-10.,max=10.); model.opacities.clamp_(min=-10.,max=10.)
        with torch.no_grad():
            rc2=model.render(cams[ci]).unsqueeze(0).permute(0,3,1,2)
            psnr=_psnr(rc2,gt)
        stats.append({"camera":ci,"psnr":round(psnr,3)}); print(f"Camera {ci}: PSNR={psnr:.2f}",flush=True)
    psnrs=np.array([s["psnr"] for s in stats])
    out={"schema_version":2,"phase":"C29-E","n_cameras":n,"camera_stats":stats,
        "psnr_spread":round(float(psnrs.max()-psnrs.min()),3),
        "verdict":"KEEP" if psnrs.max()-psnrs.min()>1 else "MAYBE" if psnrs.max()-psnrs.min()>0.5 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out}")
if __name__=="__main__": main()
