# VOS0 Phase 1+2: real per-Gaussian trace on frozen 30K checkpoint.
# Phase 1a visibility freq (DS=1 forward), 1b grad freq + update magnitude (DS=2, raw_params).
import argparse, json, math, time, os, gc
import numpy as np
import torch
import vos0_lib as V

def train_cams(cams_json):
    return [c for i,c in enumerate(cams_json) if i % 8 != 0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--h_trace", type=int, default=800)
    ap.add_argument("--h_grad", type=int, default=150)
    ap.add_argument("--ds_trace", type=int, default=1)
    ap.add_argument("--ds_grad", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev="cuda"; scene=args.scene; t0=time.time()
    gs=V.load_ply(f"data/official/mipnerf360/{scene}/point_cloud.ply")
    N=gs["n"]
    cams_json=json.load(open(f"data/official/mipnerf360/{scene}/cameras.json"))
    tcams=train_cams(cams_json)
    cams=[V.build_camera({**c,"width":c["width"]//args.ds_trace,"height":c["height"]//args.ds_trace}) for c in tcams]
    nC=len(cams)
    print(f"[{scene}] N={N} nCams={nC} ds_trace={args.ds_trace} ds_grad={args.ds_grad} h_trace={args.h_trace} h_grad={args.h_grad}", flush=True)

    # ---- Phase 1a: visibility frequency (forward-only, DS=1) ----
    vis_counts=np.zeros(N,dtype=np.int64)
    rng=np.random.default_rng(args.seed)
    t1=time.time()
    for it in range(args.h_trace):
        ci=int(rng.integers(nC))
        out=V.render(cams[ci], gs, active_degree=3)
        vis_counts+=out["visibility"].detach().cpu().numpy().astype(np.int64)
        del out
        if it%200==0: torch.cuda.empty_cache()
    vis_freq=vis_counts/float(args.h_trace)
    print(f"[{scene}] visibility done {time.time()-t1:.1f}s mean_vis={vis_freq.mean():.4f}", flush=True)

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    # ---- Phase 1b: gradient frequency + update magnitude (ds_grad, real Adam) ----
    cg=[V.build_camera({**c,"width":c["width"]//args.ds_grad,"height":c["height"]//args.ds_grad}) for c in tcams]
    raw={k:torch.tensor(gs[k],device=dev,dtype=torch.float32,requires_grad=True)
         for k in ["xyz","f_dc","f_rest","opacity","scaling","rotation"]}
    # group params for optimizer: xyz, f_dc, f_rest, opacity, scaling, rotation
    opt=torch.optim.Adam([{"params":[raw[k]],"lr":args.lr} for k in raw], betas=(0.9,0.999), eps=1e-15)
    grad_counts={k:np.zeros(N,dtype=np.int64) for k in raw}
    upd_mag_sq=np.zeros(N,dtype=np.float64)
    upd_steps=np.zeros(N,dtype=np.int64)
    t2=time.time()
    for it in range(args.h_grad):
        ci=int(rng.integers(nC))
        out=V.render(cg[ci], gs, active_degree=3, raw_params=raw, use_precomp=True)
        img=out["img"]; loss=(img*img).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        for k,v in raw.items():
            if v.grad is not None:
                g=v.grad.detach()
                gn=g.norm(dim=tuple(range(1,g.dim()))) if g.dim()>1 else g.abs()
                grad_counts[k]+=(gn>0).cpu().numpy().astype(np.int64)
        pre={k:v.detach().clone() for k,v in raw.items()}
        opt.step()
        with torch.no_grad():
            step_has=np.zeros(N,dtype=bool)
            for k,v in raw.items():
                d=(v-pre[k]).detach()
                dn=d.norm(dim=tuple(range(1,d.dim()))) if d.dim()>1 else d.abs()
                upd_mag_sq+=(dn.pow(2)).cpu().numpy().astype(np.float64)
                step_has|=((dn>0).cpu().numpy())
            upd_steps+=step_has.astype(np.int64)
        del out,img
        if it%50==0: torch.cuda.empty_cache()
    active_params=[k for k in raw if k!="f_rest"]  # f_rest has no grad in precomp mode
    grad_freq_mean=np.mean([grad_counts[k]/float(args.h_grad) for k in active_params],axis=0)
    upd_mag_per_step=np.sqrt(upd_mag_sq/np.maximum(upd_steps,1))
    print(f"[{scene}] grad/update done {time.time()-t2:.1f}s grad_freq_mean={grad_freq_mean.mean():.4f} upd_mag={upd_mag_per_step.mean():.3e}", flush=True)

    # ---- Phase 2: partition ----
    labels=np.where(vis_freq>0.5,"hot",np.where(vis_freq>=0.1,"warm","cold"))
    def pstats(mask):
        if mask.sum()==0: return {"count":0,"frac":0.0}
        return {"count":int(mask.sum()),"frac":float(mask.mean()),
                "vis_freq_mean":float(vis_freq[mask].mean()),
                "vis_freq_p10":float(np.percentile(vis_freq[mask],10)),
                "vis_freq_p50":float(np.percentile(vis_freq[mask],50)),
                "vis_freq_p90":float(np.percentile(vis_freq[mask],90)),
                "grad_freq_mean":float(grad_freq_mean[mask].mean()),
                "upd_mag_per_step":float(upd_mag_per_step[mask].mean()),
                "upd_steps_mean":float(upd_steps[mask].mean()),
                "update_budget_frac":float((grad_freq_mean[mask]*upd_mag_per_step[mask]).sum()
                                            /max((grad_freq_mean*upd_mag_per_step).sum(),1e-30))}
    partition={name:pstats(labels==name) for name in ["hot","warm","cold"]}
    hist,_=np.histogram(vis_freq,bins=20,range=(0.0,1.0))
    out=dict(scene=scene,N=int(N),n_cams=int(nC),ds_trace=int(args.ds_trace),ds_grad=int(args.ds_grad),
        seed=args.seed,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        h_trace=int(args.h_trace),h_grad=int(args.h_grad),lr=args.lr,wall_s=time.time()-t0,
        visibility=dict(mean=float(vis_freq.mean()),
            p10=float(np.percentile(vis_freq,10)),p50=float(np.percentile(vis_freq,50)),
            p90=float(np.percentile(vis_freq,90)),min=float(vis_freq.min()),max=float(vis_freq.max()),
            hist_20bins=(hist/hist.sum()).tolist()),
        grad_freq_mean_all=float(grad_freq_mean.mean()),
        update_mag_rms_per_step_all=float(upd_mag_per_step.mean()),
        partition=partition,
        partition_thresholds=dict(hot="vis>0.5",warm="0.1<=vis<=0.5",cold="vis<0.1"))
    outpath=args.out or f"_vos0_trace_{scene}.json"
    json.dump(out,open(outpath,"w"),indent=2)
    # persist per-Gaussian arrays for the Phase-3 ablation
    base=outpath[:-5] if outpath.endswith(".json") else outpath
    np.save(base+"_coldmask.npy", (labels=="cold"))
    np.save(base+"_visfreq.npy", vis_freq)
    np.save(base+"_gradfreq.npy", grad_freq_mean)
    print(f"[{scene}] wrote {outpath}",flush=True)
    print(f"[{scene}] cold={partition['cold']['frac']:.4f} warm={partition['warm']['frac']:.4f} hot={partition['hot']['frac']:.4f}",flush=True)

if __name__=="__main__":
    main()
