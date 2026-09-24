#!/usr/bin/env python3
"""C31-IJ: Convergence Pattern & Eval Frequency — measure loss trajectory.

Characterizes per-iteration improvement rate to determine at what point
convergence slows and when eval/full-train cycles can be spaced out.
"""
from __future__ import annotations
import argparse,gc,json,math,sys,time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT))
from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--gpu",type=int,default=0)
    args=p.parse_args()
    device=f"cuda:{args.gpu}"; torch.cuda.set_device(device); torch.manual_seed(42); np.random.seed(42)
    dataset=GTDataset(scene="room",repo_root=ROOT,resolution="1080p",device=device)
    sfm=load_initial_checkpoint("room",ROOT,device=device)
    model=GaussianModel(num_points=sfm["xyz"].shape[0],sh_degree=0,max_sh_degree=3,device=device)
    model.init_from_sfm(xyz=sfm["xyz"],
        opacity_logit=torch.logit(torch.full((sfm["xyz"].shape[0],1),0.1,device=device)),
        scales_log=sfm.get("scales"),rotations_raw=sfm.get("rotations"),shs=sfm.get("shs"))
    sls=float(sfm["xyz"].norm(dim=-1).max().item()); print(f"Gs: {model.xyz.shape[0]:,}",flush=True)
    opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},
        {"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)

    losses=[]; psnrs=[]
    for step in range(args.steps):
        ci=step%len(dataset); camera=dataset.get_camera(ci); gt=dataset.get_gt_image(ci)
        nd=min(3,step//500)
        if nd!=model.sh_degree: model.set_sh_degree(nd); opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)
        data=model.forward()
        r,_,_=rasterization(means=data["xyz"],quats=data["rotations"],scales=data["scales"],opacities=data["opacity"],colors=data["shs"],viewmats=camera.viewmatrix.unsqueeze(0),Ks=camera.K.unsqueeze(0),width=camera.image_width,height=camera.image_height,tile_size=16,packed=True,sh_degree=model.sh_degree,radius_clip=0.,eps2d=0.1,render_mode="RGB")
        rendered=r[0].clamp(0,1)
        loss_dict=combined_loss(rendered,gt,lambda_dssim=0.2); loss=loss_dict["loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.cuda.synchronize()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        opt.step(); torch.cuda.synchronize()
        if step>=200 and step%100==0: model.densification(grad_threshold=2e-4); opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)
        if step>=200 and step%100==0: model.prune(opacity_threshold=0.005); opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)
        with torch.no_grad():
            mse=torch.mean((rendered-gt)**2).item(); psnr=10*math.log10(1/max(mse,1e-10))
        losses.append(loss.item()); psnrs.append(psnr)
        if step%100==0 or step==args.steps-1: print(f"  Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} N={model.xyz.shape[0]:,}",flush=True)

    # Convergence analysis
    l=np.array(losses); p=np.array(psnrs)
    # Per-window improvement rate
    windows={"first_100":(0,100),"100_200":(100,200),"200_300":(200,300),"300_500":(300,500),"500_1000":(500,1000)}
    conv={}
    for name,(s,e) in windows.items():
        if e>len(l): e=len(l)
        if s>=e: continue
        window_losses=l[s:e]
        improvement_rate=-np.mean(np.diff(window_losses))
        conv[name]={
            "mean_loss":float(np.mean(window_losses)),
            "final_loss":float(window_losses[-1]),
            "mean_psnr":float(np.mean(p[s:e])),
            "improvement_per_step":float(improvement_rate),
            "fraction_of_initial":float(window_losses[-1]/max(l[0],1e-10))
        }

    summary={"schema_version":2,"phase":"C31-IJ","n_steps":args.steps,
        "convergence":conv,
        "early_vs_late_improvement":conv.get("first_100",{}).get("improvement_per_step",0)/max(conv.get("500_1000",{}).get("improvement_per_step",1e-10),1e-10),
        "final_psnr":float(p[-1]),"final_gaussians":model.xyz.shape[0],
        "recommendation":"If improvement rate drops >10x after 200 steps, eval every N steps instead of every step saves N-1/N cost",
        "verdict":"SCREENING"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(summary,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")

if __name__=="__main__": main()
