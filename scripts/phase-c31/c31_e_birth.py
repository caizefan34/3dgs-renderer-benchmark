#!/usr/bin/env python3
"""C31-E: Gaussian Birth Optimization State — measure newborn Gaussian transients.
Compare new Gaussians vs same-age ordinary Gaussians over 100 post-birth steps.
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
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True); p.add_argument("--steps",type=int,default=500)
    p.add_argument("--scene",default="room"); p.add_argument("--gpu",type=int,default=0)
    args=p.parse_args()
    device=f"cuda:{args.gpu}"; torch.cuda.set_device(device)
    torch.manual_seed(42); np.random.seed(42)
    dataset=GTDataset(scene=args.scene,repo_root=ROOT,resolution="1080p",device=device)
    sfm=load_initial_checkpoint(args.scene,ROOT,device=device)
    model=GaussianModel(num_points=sfm["xyz"].shape[0],sh_degree=0,max_sh_degree=3,device=device)
    model.init_from_sfm(xyz=sfm["xyz"],
        opacity_logit=torch.logit(torch.full((sfm["xyz"].shape[0],1),0.1,device=device)),
        scales_log=sfm.get("scales"),rotations_raw=sfm.get("rotations"),shs=sfm.get("shs"))
    sls=float(sfm["xyz"].norm(dim=-1).max().item())
    print(f"Gs: {model.xyz.shape[0]:,}",flush=True)
    opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},
        {"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)

    births=[]  # track birth events + follow-up
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

        n_before=model.xyz.shape[0]
        topology_happened=False
        if step>=200 and step%100==0:
            dc=model.densification(grad_threshold=2e-4)
            if dc["cloned"]+dc["split"]>0:
                n_new=model.xyz.shape[0]-n_before
                ev={"step":step,"n_new":n_new,"new_start":n_before,"n_before":n_before,
                    "n_after":model.xyz.shape[0],"post_tracking":[],"tracking_ended_reason":None}
                births.append(ev)
                print(f"  Birth at step {step}: +{n_new} Gs (total: {model.xyz.shape[0]:,})",flush=True)
                topology_happened=True
                opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)
        if step>=200 and step%100==0:
            pruned=model.prune(opacity_threshold=0.005)
            if pruned>0:
                topology_happened=True
                opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)

        # Track births in following steps. STOP once a topology event occurs
        # after the birth: pruning shifts all following indices, so index-based
        # cohort selection is no longer valid.
        for ev in births:
            if ev["tracking_ended_reason"] is not None: continue
            if len(ev["post_tracking"])>=100:
                ev["tracking_ended_reason"]="reached 100 tracked steps"; continue
            start,n_new=ev["new_start"],ev["n_new"]
            end=start+n_new
            if end>model.xyz.shape[0]:
                ev["tracking_ended_reason"]="cohort fully pruned"; continue
            with torch.no_grad():
                new_xyz=model.xyz[start:end].norm(dim=-1).mean().item()
                new_op=torch.sigmoid(model.opacity[start:end]).mean().item()
                new_scale=torch.exp(model.scales[start:end]).mean().item()
                ev["post_tracking"].append({"step":step,"n_cohort_now":end-start,
                    "mean_xyz_norm":new_xyz,"mean_opacity":new_op,"mean_scale":new_scale})
        if topology_happened:
            for ev in births:
                if ev["tracking_ended_reason"] is None and ev["step"]<step:
                    ev["tracking_ended_reason"]="topology event after birth"

        if step%100==0:
            with torch.no_grad():
                mse=torch.mean((rendered-gt)**2).item()
                psnr=10*math.log10(1/max(mse,1e-10))
            print(f"  Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} N={model.xyz.shape[0]:,}",flush=True)

    summary={"schema_version":2,"phase":"C31-E","n_steps":args.steps,"n_birth_events":len(births),
        "birth_events":births,"verdict":"SCREENING"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(summary,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")

if __name__=="__main__": main()
