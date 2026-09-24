#!/usr/bin/env python3
"""C31-D: Parameter-Group Update Cadence — measure marginal benefit per parameter group.

Different groups may not need the same optimization cadence. Measure gradient/update
norms, relative change, and delta-loss sensitivity for each parameter group individually.
"""
from __future__ import annotations
import argparse, gc, json, math, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--steps",type=int,default=500)
    p.add_argument("--scene",default="room"); p.add_argument("--gpu",type=int,default=0)
    args = p.parse_args()
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

    param_groups=["xyz","rotations","scales","opacity","shs"]
    param_attr={"xyz":model.xyz,"rotations":model.rotations,"scales":model.scales,"opacity":model.opacity,"shs":model.shs}
    records=[]

    for step in range(args.steps):
        ci=step%len(dataset); camera=dataset.get_camera(ci); gt=dataset.get_gt_image(ci)
        nd=min(3,step//500)
        if nd!=model.sh_degree: model.set_sh_degree(nd); opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)
        data=model.forward()
        r,_,_=rasterization(means=data["xyz"],quats=data["rotations"],scales=data["scales"],opacities=data["opacity"],colors=data["shs"],viewmats=camera.viewmatrix.unsqueeze(0),Ks=camera.K.unsqueeze(0),width=camera.image_width,height=camera.image_height,tile_size=16,packed=True,sh_degree=model.sh_degree,radius_clip=0.,eps2d=0.1,render_mode="RGB")
        rendered=r[0].clamp(0,1)
        loss_dict=combined_loss(rendered,gt,lambda_dssim=0.2); loss=loss_dict["loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.cuda.synchronize()

        # Per-group stats. Re-derive param references every iteration:
        # densification/pruning replace model.xyz etc. with new tensors.
        param_attr = {g: getattr(model, g) for g in param_groups}
        rec={}
        for g in param_groups:
            p=param_attr[g]
            gn=p.grad.norm().item() if p.grad is not None else 0.
            rec[f"{g}_grad_norm"]=gn
        rec.update({"step":step,"loss":loss.item(),"psnr":10*math.log10(1/max(((rendered-gt)**2).mean().item(),1e-10)),
            "n_gaussians":model.xyz.shape[0]})

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        # Capture actual pre-step tensors (detached copies) for true delta norms
        pre_step = {g: param_attr[g].detach().clone() for g in param_groups}
        opt.step(); torch.cuda.synchronize()
        for g in param_groups:
            after=param_attr[g]
            if after.shape == pre_step[g].shape:
                rec[f"{g}_delta_norm"]=float((after - pre_step[g]).norm().item())
            else:
                rec[f"{g}_delta_norm"]=float("nan")  # topology changed this step
        del pre_step

        if step>=200 and step%100==0:
            model.densification(grad_threshold=2e-4); opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)
        if step>=200 and step%100==0:
            model.prune(opacity_threshold=0.005); opt=torch.optim.Adam([{"params":[model.xyz],"lr":1.6e-4*sls},{"params":[model.rotations],"lr":1e-3},{"params":[model.scales],"lr":5e-3},{"params":[model.opacity],"lr":5e-2},{"params":[model.shs],"lr":2.5e-3}],eps=1e-15)

        records.append(rec)
        if step%100==0 or step==args.steps-1:
            print(f"  Step {step}: loss={loss.item():.4f} PSNR={rec['psnr']:.2f} N={model.xyz.shape[0]:,}",flush=True)

    # Aggregate: early vs late mean grad/update
    n=len(records); third=n//3
    def _agg(recs,key):
        out={}
        for g in param_groups:
            vals=np.array([r[f"{g}_{key}"] for r in recs],dtype=np.float64)
            out[g]=float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
        return out
    early,mid,late=records[:third],records[third:2*third],records[2*third:]
    summary={"schema_version":2,"phase":"C31-D","n_steps":args.steps,
        "param_group_stats":{g:{"early_grad":_agg(early,"grad_norm")[g],"mid_grad":_agg(mid,"grad_norm")[g],
        "late_grad":_agg(late,"grad_norm")[g],"early_delta_norm":_agg(early,"delta_norm")[g],
        "mid_delta_norm":_agg(mid,"delta_norm")[g],"late_delta_norm":_agg(late,"delta_norm")[g]} for g in param_groups},
        "gradient_spread":(lambda v: (max(v)/max(min((x for x in v if x and x>0),default=1.0),1e-10) if v else float("nan")))
            ([v for v in _agg(records,"grad_norm").values() if v is not None and not np.isnan(v)]),
        "notes":"delta_norm is NaN on topology-change steps; nanmean applied. Grad captured AFTER loss.backward, BEFORE clip. Re-derived param refs each step (densification replaces tensors).",
        "verdict":"SCREENING"}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(summary,open(args.out,"w"),indent=2)
    print(f"Saved {args.out}")

if __name__=="__main__": main()
