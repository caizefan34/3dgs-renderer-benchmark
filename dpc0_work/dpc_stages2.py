#!/usr/bin/env python3
"""DPC0 protocol stages: directional, one-step, timing. Imports dpc_protocol infra."""
import sys, os, json, math, datetime, time
import numpy as np
import torch
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpc_protocol import (load_dpc, get_cameras, get_lrs, render_loss, grads_of,
                          opt_vec, pair_metrics, GRAD_ORDER, GROUP_LR, nowz, sha_file)

CELLS = [("room",10000),("room",15000),("bicycle",10000),("bicycle",15000)]
DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")

def save(scene, it, obj):
    p = f"/root/epic05_dpc0/runs/{scene}_{it}_{obj.__class__.__name__ if False else ''}"
    path = f"/root/epic05_dpc0/runs/{scene}_{it}.json"
    d = {}
    if os.path.exists(path):
        try: d = json.load(open(path))
        except Exception: d = {}
    d[obj.get("_stage","x")] = obj
    json.dump(d, open(path,"w"), indent=2, default=str)

# ------------------------------------------------------------- directional
def stage_directional(scene, it, gpu, n_dirs=16, steps=(1e-3,3e-4,1e-4), seed=20260930):
    torch.cuda.set_device(gpu); torch.manual_seed(0)
    exp, dpc, orig_bwd = load_dpc()
    ckpt = C.load_checkpoint(scene, it); model = C.build_model(ckpt); C.scene_reset()
    ds, source, peer, holdout, n = get_cameras(scene)
    lrs = get_lrs(model)
    params = [getattr(model, nm) for nm in GRAD_ORDER]
    fused = C.make_fused_loss()
    ci = source[0]; cam, gt = ds.get_item(ci)
    theta0 = [p.detach().clone() for p in params]
    lr_t = {nm: lrs[GROUP_LR[nm]] for nm in GRAD_ORDER}

    # DPC gradient at theta0 (deterministic)
    exp.higs_rasterize_backward = dpc.higs_rasterize_backward
    L0 = render_loss(model, cam, gt, fused)
    gd = grads_of(L0, params)
    torch.cuda.synchronize()
    # analytic directional derivative per direction: dot(grad*lr, d_hat)
    gv = {nm: gd[nm].astype(np.float64)*lr_t[nm] for nm in GRAD_ORDER}

    # frozen Rademacher directions in optimizer-unit coords
    rng = np.random.RandomState(seed)
    dirs = []
    for _ in range(n_dirs):
        du = {nm: (rng.randint(0,2,size=gd[nm].shape)*2-1).astype(np.float64) for nm in GRAD_ORDER}
        nrm = math.sqrt(sum(float((du[nm]**2).sum()) for nm in GRAD_ORDER))
        du = {nm: du[nm]/nrm for nm in GRAD_ORDER}
        dirs.append(du)

    def loss_with(delta):
        for nm in GRAD_ORDER:
            getattr(model,nm).data.copy_(theta0[GRAD_ORDER.index(nm)] + torch.as_tensor(delta[nm], dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()
        L = render_loss(model, cam, gt, fused)
        return float(L.detach().cpu())

    rel_errs = []
    for di, du in enumerate(dirs):
        # analytic
        analytic = sum(float((gv[nm]*du[nm]).sum()) for nm in GRAD_ORDER)
        step_errs = []
        for h in steps:
            delta = {nm: h*du[nm]*lr_t[nm] for nm in GRAD_ORDER}
            lp = loss_with(+delta); lm = loss_with(-delta)
            numeric = (lp - lm)/(2*h)
            denom = abs(numeric) if abs(numeric) > 1e-12 else 1e-30
            step_errs.append(abs(analytic-numeric)/denom)
        rel = float(np.median(step_errs))
        rel_errs.append({"dir":di,"rel_err":rel,"step_errs":step_errs,
                         "analytic":analytic,"degenerate":bool(np.isfinite(rel) and rel<1e-6)})
    # restore theta0
    for i,nm in enumerate(GRAD_ORDER):
        getattr(model,nm).data.copy_(theta0[i])
    arr = np.array([r["rel_err"] for r in rel_errs])
    finite = arr[np.isfinite(arr)]
    nondeg = float(np.mean([r["degenerate"] for r in rel_errs]))
    out = {"_stage":"directional","cell":f"{scene}@{it}","gpu":gpu,"cam":ci,
           "n_dirs":n_dirs,"steps":list(steps),"seed":seed,
           "rel_err_median":float(np.median(finite)) if len(finite) else None,
           "rel_err_p90":float(np.percentile(finite,90)) if len(finite) else None,
           "n_degenerate_frac":nondeg,"conclusive":bool(nondeg<=0.25),
           "per_dir":rel_errs}
    return out

# ------------------------------------------------------------- one-step Adam
def _psnr(img, gt):
    mse = float(((img-gt)**2).mean())
    return 100.0*math.log10(1.0/max(mse,1e-12))

def stage_onestep(scene, it, gpu, n_c0=8, seed=20260930):
    torch.cuda.set_device(gpu); torch.manual_seed(0)
    exp, dpc, orig_bwd = load_d0pc() if False else load_dpc()
    ckpt = C.load_checkpoint(scene, it); model = C.build_model(ckpt); C.scene_reset()
    ds, source, peer, holdout, n = get_cameras(scene)
    params = [getattr(model, nm) for nm in GRAD_ORDER]
    fused = C.make_fused_loss()
    theta0 = [p.detach().clone() for p in params]
    opt_sd = json.loads(json.dumps({}))  # placeholder
    import copy
    opt_state = copy.deepcopy(model.optimizer.state_dict())

    def restore():
        for i,nm in enumerate(GRAD_ORDER):
            getattr(model,nm).data.copy_(theta0[i])
        model.optimizer.load_state_dict(copy.deepcopy(opt_state))
        model.optimizer.zero_grad(set_to_none=True)

    def evaluate():
        torch.cuda.synchronize()
        # source loss (source[0])
        cs, gs = ds.get_item(source[0]); Ls = float(render_loss(model, cs, gs, fused).detach().cpu())
        # peer mean loss
        peer_losses = []
        for pi in peer:
            cp, gp = ds.get_item(pi); peer_losses.append(float(render_loss(model, cp, gp, fused).detach().cpu()))
        peer_mean = float(np.mean(peer_losses))
        # holdout PSNR
        psnrs = []
        for hi in holdout:
            ch, gh = ds.get_item(hi)
            img,_,_ = C.render_view(model, ch, model.active_sh_degree)
            torch.cuda.synchronize()
            psnrs.append(_psnr(img.clamp(0,1).cpu().numpy(), gh.cpu().numpy()))
        return {"source_loss":Ls,"peer_mean_loss":peer_mean,
                "holdout_psnr_median":float(np.median(psnrs)),
                "holdout_psnr_worst":float(np.min(psnrs)),"psnrs":psnrs}

    def run_arm(bwd_fn):
        restore()
        L0 = render_loss(model, *ds.get_item(source[0]), fused)
        L0_val = float(L0.detach().cpu())
        g = torch.autograd.grad(L0, params, retain_graph=True, allow_unused=True)
        for nm,gr in zip(GRAD_ORDER,g):
            getattr(model,nm).grad = None if gr is None else gr.detach().clone()
        model.optimizer.step()
        torch.cuda.synchronize()
        ev = evaluate()
        ev["L0_source"] = L0_val
        ev["source_improvement"] = L0_val - ev["source_loss"]
        return ev

    # C0 envelope: n_c0 independent runs
    exp.higs_rasterize_backward = orig_bwd
    c0_runs = [run_arm(orig_bwd) for _ in range(n_c0)]
    # DPC arm
    exp.higs_rasterize_backward = dpc.higs_rasterize_backward
    dpc_run = run_arm(dpc.higs_rasterize_backward)
    exp.higs_rasterize_backward = orig_bwd
    restore()

    def env(key, lower_is_better=True):
        vals = [r[key] for r in c0_runs]
        return float(np.min(vals)), float(np.max(vals))
    s_lo, s_hi = env("source_loss")
    p_lo, p_hi = env("peer_mean_loss")
    ps_lo, ps_hi = env("holdout_psnr_median")
    pw_lo, pw_hi = env("holdout_psnr_worst")
    c0_impr = float(np.mean([r["source_improvement"] for r in c0_runs]))
    src_tol = 0.001*abs(c0_impr)
    peer_tol = 0.002*abs(float(np.mean([r["peer_mean_loss"] for r in c0_runs])))
    psnr_med_tol = 0.01; psnr_worst_tol = 0.03
    checks = {
        "source_loss": {"dpc":dpc_run["source_loss"],"c0_min":s_lo,"c0_max":s_hi,"tol":src_tol,
                        "pass":bool(s_lo-src_tol <= dpc_run["source_loss"] <= s_hi+src_tol)},
        "peer_mean_loss": {"dpc":dpc_run["peer_mean_loss"],"c0_min":p_lo,"c0_max":p_hi,"tol":peer_tol,
                        "pass":bool(p_lo-peer_tol <= dpc_run["peer_mean_loss"] <= p_hi+peer_tol)},
        "holdout_psnr_median": {"dpc":dpc_run["holdout_psnr_median"],"c0_min":ps_lo,"c0_max":ps_hi,"tol":psnr_med_tol,
                        "pass":bool(ps_lo-psnr_med_tol <= dpc_run["holdout_psnr_median"] <= ps_hi+psnr_med_tol)},
        "holdout_psnr_worst": {"dpc":dpc_run["holdout_psnr_worst"],"c0_min":pw_lo,"c0_max":pw_hi,"tol":psnr_worst_tol,
                        "pass":bool(pw_lo-psnr_worst_tol <= dpc_run["holdout_psnr_worst"] <= pw_hi+psnr_worst_tol)},
    }
    all_pass = all(c["pass"] for c in checks.values())
    out = {"_stage":"onestep","cell":f"{scene}@{it}","gpu":gpu,"n_c0":n_c0,
           "c0_runs":c0_runs,"dpc_run":dpc_run,"checks":checks,
           "c0_source_improvement_mean":c0_impr,"pass":bool(all_pass)}
    return out

# ------------------------------------------------------------- timing
def stage_timing(scene, it, gpu, n_rep=128, n_warm=10):
    torch.cuda.set_device(gpu); torch.manual_seed(0)
    exp, dpc, orig_bwd = load_dpc()
    ckpt = C.load_checkpoint(scene, it); model = C.build_model(ckpt); C.scene_reset()
    ds, source, peer, holdout, n = get_cameras(scene)
    params = [getattr(model, nm) for nm in GRAD_ORDER]
    fused = C.make_fused_loss()
    ci = source[0]; cam, gt = ds.get_item(ci)
    L = render_loss(model, cam, gt, fused)  # build graph once
    torch.cuda.synchronize()

    def bench(bwd_fn, reps):
        torch.cuda.synchronize()
        for _ in range(n_warm):
            g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
            for nm,gr in zip(GRAD_ORDER,g):
                getattr(model,nm).grad = None if gr is None else gr.detach().clone()
        torch.cuda.synchronize()
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        times = []
        for _ in range(reps):
            start.record()
            g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
            for nm,gr in zip(GRAD_ORDER,g):
                getattr(model,nm).grad = None if gr is None else gr.detach().clone()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return times

    exp.higs_rasterize_backward = orig_bwd
    c0_t = bench(orig_bwd, n_rep)
    exp.higs_rasterize_backward = dpc.higs_rasterize_backward
    dpc_t = bench(dpc.higs_rasterize_backward, n_rep)
    exp.higs_rasterize_backward = orig_bwd
    c0_med = float(np.median(c0_t)); dpc_med = float(np.median(dpc_t))
    # peak VRAM
    torch.cuda.reset_peak_memory_stats(gpu)
    g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(gpu)/ (1024**2)
    out = {"_stage":"timing","cell":f"{scene}@{it}","gpu":gpu,"cam":ci,"n_rep":n_rep,
           "c0_bwd_ms_median":c0_med,"c0_bwd_ms_mean":float(np.mean(c0_t)),
           "dpc_total_ms_median":dpc_med,"dpc_total_ms_mean":float(np.mean(dpc_t)),
           "backward_speedup":c0_med/dpc_med if dpc_med>0 else None,
           "peak_vram_mb":peak,
           "c0_samples_ms":c0_t,"dpc_samples_ms":dpc_t}
    return out

STAGES = {"directional":stage_directional,"onestep":stage_onestep,"timing":stage_timing}
if __name__ == "__main__":
    stage = sys.argv[1]; scene = sys.argv[2]; it = int(sys.argv[3]); gpu = int(sys.argv[4]) if len(sys.argv)>4 else 0
    out = STAGES[stage](scene, it, gpu)
    print(json.dumps(out, indent=2, default=str))
