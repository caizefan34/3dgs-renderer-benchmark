#!/usr/bin/env python3
"""C29-J: Multi-GPU gradient/state communication overhead estimate."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
from _c29_training import GaussianModel, W, H, DEV

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True)
    args=p.parse_args()
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    model=GaussianModel(scene,3,DEV)
    total=sum(p.numel()*4/(1024**2) for p in [model.means,model.quats,model.scales,model.opacities,model.shs])
    pcie_bw=32e9/8; nvlink_bw=600e9/8; data_bytes=total*1024*1024*2
    cpcie=data_bytes/pcie_bw*1000; cnvlink=data_bytes/nvlink_bw*1000
    titer=8.0
    out={"schema_version":2,"phase":"C29-J","total_params_mb":round(total,2),"pcie_allreduce_ms":round(cpcie,3),
        "nvlink_allreduce_ms":round(cnvlink,3),"comm_pct_pcie":round(cpcie/titer*100,1),"comm_pct_nvlink":round(cnvlink/titer*100,1),
        "verdict":"KEEP" if cpcie/titer>0.1 else "MAYBE" if cpcie/titer>0.05 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out}")
if __name__=="__main__": main()
