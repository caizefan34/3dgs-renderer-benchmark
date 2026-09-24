#!/usr/bin/env python3
"""C31-GH: Camera Utility & Redundancy — measure per-camera gradient contribution.
Higher redundancy = more candidates for sparse camera scheduling.
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
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--max-cameras",type=int,default=50)  # sample 50 cameras for speed
    args=p.parse_args()
    device=f"cuda:{args.gpu}"; torch.cuda.set_device(device); torch.manual_seed(42); np.random.seed(42)
    dataset=GTDataset(scene="room",repo_root=ROOT,resolution="1080p",device=device)
    sfm=load_initial_checkpoint("room",ROOT,device=device)
    model=GaussianModel(num_points=sfm["xyz"].shape[0],sh_degree=0,max_sh_degree=3,device=device)
    model.init_from_sfm(xyz=sfm["xyz"],
        opacity_logit=torch.logit(torch.full((sfm["xyz"].shape[0],1),0.1,device=device)),
        scales_log=sfm.get("scales"),rotations_raw=sfm.get("rotations"),shs=sfm.get("shs"))
    print(f"Gs: {model.xyz.shape[0]:,}",flush=True)
    opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*float(sfm["xyz"].norm(dim=-1).max().item())},
        {"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},
        {"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)

    n_cam=min(len(dataset),args.max_cameras)
    camera_contribs=[]
    for ci in range(n_cam):
        camera=dataset.get_camera(ci); gt=dataset.get_gt_image(ci)
        data=model.forward()
        r,_,_=rasterization(means=data["xyz"],quats=data["rotations"],scales=data["scales"],opacities=data["opacity"],colors=data["shs"],viewmats=camera.viewmatrix.unsqueeze(0),Ks=camera.K.unsqueeze(0),width=camera.image_width,height=camera.image_height,tile_size=16,packed=True,sh_degree=0,radius_clip=0.,eps2d=0.1,render_mode="RGB")
        rendered=r[0].clamp(0,1)
        loss=combined_loss(rendered,gt,lambda_dssim=0.2)["loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.cuda.synchronize()
        total_grad_norm=sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
        pos_grad_norm=model.xyz.grad.norm().item()
        camera_contribs.append({"camera_idx":int(ci),"total_grad_norm":total_grad_norm,"pos_grad_norm":pos_grad_norm,
            "loss":loss.item()})
        if ci%10==0: print(f"  Camera {ci}/{n_cam}: loss={loss.item():.4f} grad={total_grad_norm:.4f}",flush=True)
        for p in model.parameters(): p.grad=None
        gc.collect()

    norms=[c["total_grad_norm"] for c in camera_contribs]
    pos_norms=[c["pos_grad_norm"] for c in camera_contribs]
    losses=[c["loss"] for c in camera_contribs]

    # Analyze redundancy: what fraction of cameras contribute <20%/50%/80% of max gradient?
    max_norm=max(norms) if norms else 1.0
    frac_below={f"{pct}pct_max":float(sum(1 for n in norms if n<max_norm*pct/100)/len(norms))
        for pct in [10,20,30,50]}
    # Loss similarity: cameras with near-identical loss
    losses_arr=np.array(losses)
    loss_std=float(np.std(losses_arr))
    loss_mean=float(np.mean(losses_arr))
    # Correlation between camera idx and contribution (angular diversity)
    corr_with_idx=np.corrcoef(list(range(len(norms))),norms)[0,1] if len(norms)>2 else 0.0

    summary={"schema_version":2,"phase":"C31-GH","n_cameras_sampled":n_cam,
        "gradient_contribution":{"mean":float(np.mean(norms)),"median":float(np.median(norms)),
            "std":float(np.std(norms)),"max":float(max(norms)),"min":float(min(norms))},
        "position_gradient":{"mean":float(np.mean(pos_norms)),"median":float(np.median(pos_norms))},
        "redundancy_metrics":{
            "fraction_below_10pct_of_max":frac_below["10pct_max"],
            "fraction_below_20pct_of_max":frac_below["20pct_max"],
            "fraction_below_30pct_of_max":frac_below["30pct_max"],
            "fraction_below_50pct_of_max":frac_below["50pct_max"],
            "loss_std":loss_std,"loss_cv":loss_std/max(loss_mean,1e-10),
            "correlation_idx_vs_grad":corr_with_idx},
        "top_contributors":sorted(camera_contribs,key=lambda x:-x["total_grad_norm"])[:5],
        "bottom_contributors":sorted(camera_contribs,key=lambda x:x["total_grad_norm"])[:5],
        "verdict":"SCREENING"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(summary,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")

if __name__=="__main__": main()
