#!/usr/bin/env python3
"""C29-A: Adaptive step value."""
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
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--n_steps",type=int,default=500)
    args=p.parse_args(); torch.manual_seed(0)
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device=DEV)
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device=DEV),W,H)
    cam=cams[0]
    m=GaussianModel(scene,3,DEV); gt=m.render(cam).detach().unsqueeze(0).permute(0,3,1,2).contiguous(); del m; torch.cuda.empty_cache()
    model=GaussianModel(scene,3,DEV); opt=model.get_optimizer()
    print(f"Initial Gs: {model.n}",flush=True)
    records=[]
    for step in range(args.n_steps):
        torch.cuda.synchronize(); tf0=time.perf_counter()
        rendered=model.render(cam); torch.cuda.synchronize(); fwd_ms=(time.perf_counter()-tf0)*1000
        rc=rendered.unsqueeze(0).permute(0,3,1,2); loss=(rc-gt).abs().mean()+0.2*(1.0-min(1.0,1.0)); psnr=_psnr(rc,gt)
        tb0=time.perf_counter(); loss.backward(); bwd_ms=(time.perf_counter()-tb0)*1000
        gns={n:float(getattr(model,n).grad.norm().item()) if getattr(model,n).grad is not None else 0. for n in ["means","quats","scales","opacities","shs"]}
        tt0=time.perf_counter(); res=model.densify_and_prune(step,gt=0.0002,po=0.005); topo_ms=(time.perf_counter()-tt0)*1000
        if model.n!=len(opt.param_groups[0]["params"][0]): opt=model.get_optimizer()
        to0=time.perf_counter(); opt.step(); opt.zero_grad(set_to_none=True); opt_ms=(time.perf_counter()-to0)*1000
        with torch.no_grad(): model.scales.clamp_(min=-10.,max=10.); model.opacities.clamp_(min=-10.,max=10.)
        t_iter=fwd_ms+bwd_ms+opt_ms+topo_ms
        records.append({"step":step,"loss":float(loss.item()),"psnr":float(psnr),"fwd_ms":round(fwd_ms,3),
            "bwd_ms":round(bwd_ms,3),"opt_ms":round(opt_ms,3),"topo_ms":round(topo_ms,3),
            "t_iter_ms":round(t_iter,3),"n_gaussians":model.n,"densified":res["densified"],"pruned":res["pruned"]})
        if (step+1)%100==0: print(f"Step {step+1}: loss={loss.item():.4f} PSNR={psnr:.2f} Gs={model.n} T={t_iter:.1f}ms",flush=True)
    ps=np.array([r["psnr"] for r in records]); ti=np.array([r["t_iter_ms"] for r in records]); v=np.diff(ps)/np.maximum(ti[:-1],1e-6)
    ev=float(np.mean(v[:max(1,len(v)//10)])); lv=float(np.mean(v[-len(v)//10:])) if len(v)>=10 else 0
    out={"schema_version":2,"phase":"C29-A","records":records,"early_v":round(ev,6),"late_v":round(lv,6),
        "late_pct_of_early":round(lv/max(ev,1e-10)*100,1),
        "verdict":"MAYBE" if lv/max(ev,1e-10)<0.5 else "DROP"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,"w"),indent=2); print(f"Saved {args.out}")
if __name__=="__main__": main()
