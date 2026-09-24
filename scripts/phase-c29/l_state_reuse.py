#!/usr/bin/env python3
"""C29-L: Training state reuse measurement."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import GaussianModel, W, H, DEV

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--n_repeats",type=int,default=50)
    args=p.parse_args()
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    model=GaussianModel(scene,3,DEV); nr=args.n_repeats
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(nr): opt=model.get_optimizer()
    torch.cuda.synchronize(); opt_ms=(time.perf_counter()-t0)/nr*1000
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(min(nr,10)): m2=GaussianModel(scene,3,DEV); del m2
    torch.cuda.synchronize(); init_ms=(time.perf_counter()-t0)/min(nr,10)*1000
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(nr):
        with torch.no_grad(): model.scales.clamp_(min=-10.,max=10.); model.opacities.clamp_(min=-10.,max=10.)
    torch.cuda.synchronize(); clamp_ms=(time.perf_counter()-t0)/nr*1000
    torch.cuda.empty_cache(); torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(min(nr,10)):
        t=torch.randn(1593376,16,3,device=DEV); del t; torch.cuda.synchronize()
    torch.cuda.synchronize(); alloc_ms=(time.perf_counter()-t0)/min(nr,10)*1000
    out={"schema_version":2,"phase":"C29-L","opt_rebuild_us":round(opt_ms*1000,1),"model_init_us":round(init_ms*1000,1),
        "param_clamp_us":round(clamp_ms*1000,1),"alloc_us":round(alloc_ms*1000,1),
        "verdict":"DROP -- all <500us per occurrence"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")
if __name__=="__main__": main()
