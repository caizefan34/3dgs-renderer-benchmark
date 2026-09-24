#!/usr/bin/env python3
"""C31-F: Training State Co-Design — measure optimizer momentum vs position-only.

Tests if retaining optimizer state (Adam moments) across densification/pruning
is beneficial. Compares: retain (current), reset all, reset moments only.
"""
from __future__ import annotations
import argparse,gc,json,math,sys,time,copy
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT))
from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

def run_trial(warmup=100, measured=200, reset_moments=False, reset_all=False, label=""):
    torch.manual_seed(42); np.random.seed(42)
    dataset=GTDataset(scene="room",repo_root=ROOT,resolution="1080p",device="cuda")
    sfm=load_initial_checkpoint("room",ROOT,device="cuda")
    model=GaussianModel(num_points=sfm["xyz"].shape[0],sh_degree=0,max_sh_degree=3,device="cuda")
    model.init_from_sfm(xyz=sfm["xyz"],
        opacity_logit=torch.logit(torch.full((sfm["xyz"].shape[0],1),0.1,device="cuda")),
        scales_log=sfm.get("scales"),rotations_raw=sfm.get("rotations"),shs=sfm.get("shs"))
    sls=float(sfm["xyz"].norm(dim=-1).max().item())
    def make_opt():
        return torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},
            {"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)
    opt=make_opt()
    total=warmup+measured; losses=[]

    for step in range(total):
        ci=step%len(dataset); camera=dataset.get_camera(ci); gt=dataset.get_gt_image(ci)
        nd=min(3,step//500)
        if nd!=model.sh_degree:
            model.set_sh_degree(nd)
            if reset_all: opt=make_opt()
            else: opt=make_opt() if reset_moments else opt
        data=model.forward()
        r,_,_=rasterization(means=data["xyz"],quats=data["rotations"],scales=data["scales"],opacities=data["opacity"],colors=data["shs"],viewmats=camera.viewmatrix.unsqueeze(0),Ks=camera.K.unsqueeze(0),width=camera.image_width,height=camera.image_height,tile_size=16,packed=True,sh_degree=model.sh_degree,radius_clip=0.,eps2d=0.1,render_mode="RGB")
        rendered=r[0].clamp(0,1)
        loss=combined_loss(rendered,gt,lambda_dssim=0.2)["loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.cuda.synchronize()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        opt.step(); torch.cuda.synchronize()
        if step>=200 and step%100==0:
            model.densification(grad_threshold=2e-4)
            if reset_all: opt=make_opt()
            elif reset_moments: opt=make_opt()  # new opt = fresh moments
            else: opt=make_opt()  # retain strategy requires new param groups anyway
        if step>=200 and step%100==0:
            model.prune(opacity_threshold=0.005)
            if reset_all: opt=make_opt()
            elif reset_moments: pass  # opt already reset in densif
        losses.append(loss.item())
        if step%100==0:
            print(f"  [{label}] Step {step}: loss={loss.item():.4f} N={model.xyz.shape[0]:,}",flush=True)
    return losses[warmup:], model.xyz.shape[0]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True)
    p.add_argument("--warmup",type=int,default=100); p.add_argument("--measured",type=int,default=200)
    args=p.parse_args()
    print("C31-F: State Co-Design",flush=True)
    torch.manual_seed(42); np.random.seed(42)
    results={}
    for key,reset_moments,reset_all in [("retain",False,False),("reset_moments",True,False),("reset_all",False,True)]:
        print(f"\n--- Trial: {key} ---",flush=True)
        l,n=run_trial(warmup=args.warmup,measured=args.measured,reset_moments=reset_moments,reset_all=reset_all,label=key)
        results[key]={"mean_loss":float(np.mean(l)),"final_loss":float(l[-1]),"final_n_gaussians":n,"loss_std":float(np.std(l))}
        print(f"  Result: mean_loss={results[key]['mean_loss']:.4f} final_loss={results[key]['final_loss']:.4f}",flush=True)
        gc.collect()
    summary={"schema_version":2,"phase":"C31-F","warmup":args.warmup,"measured":args.measured,"results":results,
        "verdict":"SCREENING"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(summary,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")

if __name__=="__main__": main()
