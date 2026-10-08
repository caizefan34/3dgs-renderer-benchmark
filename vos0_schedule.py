# VOS0 Phase 3: cold-update schedule oracle (self-referential divergence).
# Continue-train dense vs cold-skip arms from the frozen 30K state; measure
# PSNR(dense_render, skip_render) over eval cameras + exact optimizer workload reduction.
import argparse, json, math, time, os, gc
import numpy as np
import torch
import vos0_lib as V

def psnr(a, b):
    mse = float(((a - b) ** 2).mean().item())
    if mse <= 1e-12: return 99.0
    return -10.0 * math.log10(mse)

def psnr_list(renders, refs):
    return [psnr(refs[i], renders[i]) for i in range(len(renders))]

def train_cams(cj):
    return [c for i, c in enumerate(cj) if i % 8 != 0]

def test_cams(cj):
    return [c for i, c in enumerate(cj) if i % 8 == 0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--h_iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--ds_train", type=int, default=1)
    ap.add_argument("--ds_eval", type=int, default=2)
    ap.add_argument("--n_eval", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trace", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"; scene = args.scene; t0 = time.time()
    BASE = f"data/official/mipnerf360/{scene}"
    gs = V.load_ply(f"{BASE}/point_cloud.ply"); N = gs["n"]
    cj = json.load(open(f"{BASE}/cameras.json"))

    # cold mask from Phase-1 trace
    trace_path = args.trace or f"/root/vos0_out/_vos0_trace_{scene}.json"
    cold = np.load(trace_path.replace(".json", "_coldmask.npy"))
    assert cold.shape[0] == N, f"cold mask N {cold.shape[0]} != {N}"
    cold_t = torch.as_tensor(cold, device=dev)
    always_t = ~cold_t
    cold_frac = float(cold.mean())
    print(f"[{scene}] N={N} cold_frac={cold_frac:.4f} h_iters={args.h_iters} lr={args.lr} noise={args.noise}", flush=True)

    # build train cameras (ds_train) and eval cameras (ds_eval)
    tc = train_cams(cj)
    tcams = [V.build_camera({**c, "width": c["width"]//args.ds_train, "height": c["height"]//args.ds_train}) for c in tc]
    ec = test_cams(cj)
    if args.n_eval > 0:
        ec = ec[:: max(1, len(ec)//args.n_eval)][:args.n_eval]
    eval_cams = [V.build_camera({**c, "width": c["width"]//args.ds_eval, "height": c["height"]//args.ds_eval}) for c in ec]
    nC = len(tcams)
    rng = np.random.default_rng(args.seed)

    # precompute perturbed photometric targets from the frozen 30K render
    print(f"[{scene}] precomputing targets from 30K render...", flush=True)
    targets = {}
    for ci in range(nC):
        out = V.render(tcams[ci], gs, active_degree=3, use_precomp=True)
        ref = out["img"].detach()
        noise = (torch.rand_like(ref) * 2 - 1) * args.noise
        targets[ci] = (ref + noise).clamp(0, 1)
        del out, ref, noise
        if ci % 100 == 0: torch.cuda.empty_cache()
    print(f"[{scene}] targets ready", flush=True)

    # clean frozen-30K reference renders on eval cams (GT-free quality reference)
    print(f"[{scene}] precomputing eval reference renders...", flush=True)
    eval_refs = []
    for cam in eval_cams:
        out = V.render(cam, gs, active_degree=3, use_precomp=True)
        eval_refs.append(out["img"].detach())
        del out
        if len(eval_refs) % 8 == 0: torch.cuda.empty_cache()
    print(f"[{scene}] eval refs ready ({len(eval_refs)})", flush=True)

    idx_always = torch.nonzero(always_t, as_tuple=False).squeeze(1)
    idx_cold = torch.nonzero(cold_t, as_tuple=False).squeeze(1)
    param_keys = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]

    def build_raw():
        return {k: torch.tensor(gs[k], device=dev, dtype=torch.float32, requires_grad=True) for k in param_keys}

    def split(raw):
        always = {k: raw[k][idx_always].detach().clone().requires_grad_(True) for k in param_keys}
        coldp = {k: raw[k][idx_cold].detach().clone().requires_grad_(True) for k in param_keys}
        return always, coldp

    def reconstruct(always, coldp):
        full = {}
        for k in param_keys:
            t = torch.empty(N, *always[k].shape[1:], device=dev, dtype=torch.float32)
            t = t.index_copy(0, idx_always, always[k])
            t = t.index_copy(0, idx_cold, coldp[k])
            full[k] = t
        return full

    def train_arm(k):
        raw = build_raw()
        always, coldp = split(raw)
        opt_a = torch.optim.Adam([{"params": [always[k]], "lr": args.lr} for k in param_keys], betas=(0.9, 0.999), eps=1e-15)
        opt_c = torch.optim.Adam([{"params": [coldp[k]], "lr": args.lr} for k in param_keys], betas=(0.9, 0.999), eps=1e-15)
        for it in range(args.h_iters):
            ci = int(rng.integers(nC))
            full = reconstruct(always, coldp)
            out = V.render(tcams[ci], gs, active_degree=3, raw_params=full, use_precomp=True)
            tgt = targets[ci]
            loss = ((out["img"] - tgt) ** 2).mean()
            opt_a.zero_grad(set_to_none=True); opt_c.zero_grad(set_to_none=True)
            loss.backward()
            opt_a.step()
            if (it % k) == 0:
                opt_c.step()
            del out, full, loss
            if it % 200 == 0: torch.cuda.empty_cache()
        # final params in original order
        final = reconstruct(always, coldp)
        return {k: final[k].detach() for k in param_keys}

    def eval_psnr(final_params):
        # render eval cams with final params; return list of PSNR vs dense later
        renders = []
        for ci, cam in enumerate(eval_cams):
            out = V.render(cam, gs, active_degree=3, raw_params=final_params, use_precomp=True)
            renders.append(out["img"].detach())
            del out
            if ci % 8 == 0: torch.cuda.empty_cache()
        return renders

    # dense arm (k=1) is the reference
    print(f"[{scene}] training DENSE arm...", flush=True)
    dense_final = train_arm(1)
    dense_renders = eval_psnr(dense_final)
    dense_eval_psnr = np.array(psnr_list(dense_renders, eval_refs))
    print(f"[{scene}] DENSE done (eval PSNR vs 30K ref median={np.median(dense_eval_psnr):.3f})", flush=True)

    arms = {"dense": 1, "skip2": 2, "skip4": 4, "skip8": 8}
    results = {}
    for name, k in arms.items():
        if name == "dense":
            final = dense_final; renders = dense_renders
        else:
            print(f"[{scene}] training {name} (k={k})...", flush=True)
            final = train_arm(k)
            renders = eval_psnr(final)
        # divergence: PSNR(dense_render, arm_render) per eval cam (secondary diagnostic)
        div = np.array([psnr(dense_renders[i], renders[i]) for i in range(len(renders))])
        # primary quality: PSNR loss vs clean 30K reference on held-out eval cams
        arm_eval_psnr = np.array(psnr_list(renders, eval_refs))
        loss_eval = dense_eval_psnr - arm_eval_psnr  # >0 => skip arm worse than dense
        # workload reduction: cold rows updated 1/k of iters
        workload_reduction = cold_frac * (1.0 - 1.0 / k)
        results[name] = dict(k=k,
            psnr_loss_vs_dense_eval_median=float(np.median(loss_eval)),
            psnr_loss_vs_dense_eval_mean=float(loss_eval.mean()),
            psnr_loss_vs_dense_eval_max=float(loss_eval.max()),
            psnr_loss_vs_dense_eval_per_cam=[float(x) for x in loss_eval.tolist()],
            psnr_dense_eval_median=float(np.median(dense_eval_psnr)),
            psnr_arm_eval_median=float(np.median(arm_eval_psnr)),
            psnr_divergence_vs_dense_median=float(np.median(div)),
            psnr_divergence_vs_dense_min=float(div.min()),
            psnr_divergence_vs_dense_per_cam=[float(x) for x in div.tolist()],
            workload_reduction=workload_reduction)
        print(f"[{scene}] {name}: loss_vs_dense(eval) median={np.median(loss_eval):.4f} max={loss_eval.max():.4f} divergence={np.median(div):.3f} workload_red={workload_reduction:.4f}", flush=True)
        del final, renders
        gc.collect(); torch.cuda.empty_cache()

    out = dict(scene=scene, N=int(N), cold_frac=cold_frac, h_iters=args.h_iters,
        lr=args.lr, noise=args.noise, ds_train=args.ds_train, ds_eval=args.ds_eval,
        n_eval=len(eval_cams), seed=args.seed, gpu=torch.cuda.get_device_name(0),
        wall_s=time.time() - t0, arms=results)
    outpath = args.out or f"/root/vos0_out/_vos0_sched_{scene}.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"[{scene}] wrote {outpath}", flush=True)

if __name__ == "__main__":
    main()
