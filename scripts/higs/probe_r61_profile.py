"""Round-61 profiling probe v2: windowed phase costs + union cull stats.

Replicates the 720p error-guided recipe (higs_dynamic_ts, 4 train / 3 eval
views, masked-Adam, densify every 5, anchor-densify every 2) for `steps`
steps and reports:
  - windowed means (early/mid/late) of fwd/bwd/adam/total,
  - per-event eg-refresh and densify cost,
  - direct cull-pass timings (3 consecutive calls per checkpoint),
  - union cull stats at checkpoints with a FRESH eval mask,
  - stale-vs-fresh eval wrong-prune rate (risk sizing for the prune lever).

Usage: probe_r61_profile.py [scene] [seed] [steps]
"""
import sys, os, json
sys.path.insert(0, "/root/3dgs-roadmap-matrix/benchmark")
import torch
import run_higs_train_benchmark as B
from gsplat.experimental.render.functional.gaussian_inference import (
    _HIGS_DYNAMIC_SCENE, _densify_gaussians, _prune_gaussians,
    sync_optimizer_state_for_topology_change, _cull_gaussians_batched,
)


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "mipnerf360/garden"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    W, H = 1280, 720
    NT, NE = 4, 3
    TS_RATIO = 0.35
    DENSIFY_EVERY = 5
    DENSIFY_THRESH = 0.005
    PRUNE_THRESH = 0.01
    ERR_EVERY = 25
    ERR_ALPHA = 1.0
    ERR_LAMBDA = 0.7
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(seed)

    base = "/root/epic05-data/processed"
    scene_dir = os.path.join(base, scene)
    params0 = B.load_ply_scene(os.path.join(scene_dir, "point_cloud.ply"), device)
    viewmats, Ks, train_idx, eval_idx = B.load_cameras(scene_dir, W, H, NT, NE, device)
    with open(os.path.join(scene_dir, "eval_cameras.json")) as f:
        cams = json.load(f)
    gt_dir = os.path.join(scene_dir, "eval_images")
    refs_train = B.load_reference(gt_dir, [cams[i] for i in train_idx], W, H, device)
    refs_eval = B.load_reference(gt_dir, [cams[i] for i in eval_idx], W, H, device)

    means, quats, scales, opacities, sh = [t.detach().clone() for t in params0]
    for t in (means, quats, scales, opacities, sh):
        t.requires_grad_(True)
    params = (means, quats, scales, opacities, sh)
    opt = B.make_optimizer(params, fused=True)

    dyn = _HIGS_DYNAMIC_SCENE
    dyn.reset()
    forward_fn = B.make_forward_fn(
        "higs_dynamic_ts", W, H, None, viewmats, Ks,
        radius_clip=0.0, tile_sampling_ratio=TS_RATIO,
        sampling_mode="error_guided", cull_interval=1,
        cull_cache_key="train", pixel_raster_ratio=0.35,
    )
    eval_forward_fn = B.make_forward_fn(
        "higs_dynamic_ts", W, H, None, viewmats, Ks,
        radius_clip=0.0, tile_sampling_ratio=TS_RATIO,
        sampling_mode="error_guided", cull_interval=1,
        cull_cache_key="eval", pixel_raster_ratio=0.35,
    )

    def handle():
        return getattr(dyn, "renderer_handle", None)

    def cull_slot(key):
        h = handle()
        cache = getattr(h, "_cull_cache", {})
        slot = cache.get(key)
        if slot is None or len(slot) < 2 or slot[1] is None:
            return None
        return slot[1]

    def union_stats(tag, eval_mask_override=None):
        tm = cull_slot("train")
        out = {"tag": tag, "N": int(means.shape[0])}
        if tm is None:
            return out
        em = eval_mask_override if eval_mask_override is not None else cull_slot("eval")
        out["n_train_vis"] = int(tm.sum().item())
        if em is not None:
            out["n_eval_vis"] = int(em.sum().item())
            union = tm | em
            out["n_union_vis"] = int(union.sum().item())
            out["n_union_invis"] = int((~union).sum().item())
            out["frac_union_invis"] = float((~union).float().mean().item())
            out["eval_only_vis"] = int((~tm & em).sum().item())
        return out

    # initial fresh train + eval masks
    with torch.no_grad():
        forward_fn(params, train_idx, sampling_ratio=1.0)
        eval_forward_fn(params, eval_idx, sampling_ratio=1.0)
    print("UNION0 " + json.dumps(union_stats("step0")), flush=True)

    for _ in range(3):
        with torch.no_grad():
            forward_fn(params, train_idx, sampling_ratio=TS_RATIO)
    torch.cuda.synchronize(device)

    ph = {"fwd": [], "bwd": [], "adam": [], "densify": [], "eg_refresh": [], "total": []}
    cull_meas = []
    stale_eval = None
    tile_err_cache = None
    for it in range(steps):
        is_densify_step = (it + 1) % DENSIFY_EVERY == 0
        is_anchor_step = B._is_anchor_step(True, is_densify_step, it, DENSIFY_EVERY, 2)
        step_ratio = 1.0 if is_anchor_step else TS_RATIO
        eg_mask = eg_weights = None
        if step_ratio < 1.0:
            if tile_err_cache is None or (it + 1) % ERR_EVERY == 0:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                with torch.no_grad():
                    frame_full, _, _ = forward_fn(params, train_idx, sampling_ratio=1.0)
                    tile_err_cache = B._tile_mean_errors(frame_full, refs_train, B._TILE_SIZE)
                e1.record()
                torch.cuda.synchronize(device)
                ph["eg_refresh"].append(e0.elapsed_time(e1))
            eg_mask, eg_weights = B._error_guided_mask(
                tile_err_cache, step_ratio, ERR_ALPHA, device, lambda_mix=ERR_LAMBDA,
            )
            eg_mask = eg_mask.reshape(
                tile_err_cache.shape[0], tile_err_cache.shape[1],
                tile_err_cache.shape[2],
            )
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        frame, alpha, meta = forward_fn(
            params, train_idx, sampling_ratio=step_ratio, tile_mask=eg_mask,
        )
        t1.record()
        torch.cuda.synchronize(device)
        ph["fwd"].append(t0.elapsed_time(t1))

        if it in (40, 90, 140):
            meas = []
            for _k in range(3):
                c0 = torch.cuda.Event(enable_timing=True)
                c1 = torch.cuda.Event(enable_timing=True)
                c0.record()
                _cull_gaussians_batched(
                    means.detach(), quats.detach(), scales.detach(),
                    viewmats, Ks, W, H,
                )
                c1.record()
                torch.cuda.synchronize(device)
                meas.append(c0.elapsed_time(c1))
            cull_meas.append({
                "step": it + 1, "N": int(means.shape[0]),
                "n_train_vis": int(cull_slot("train").sum().item()),
                "cull_ms": meas,
            })

        loss = B._l1_loss(frame, refs_train)
        b0 = torch.cuda.Event(enable_timing=True)
        b1 = torch.cuda.Event(enable_timing=True)
        b0.record()
        loss.backward()
        b1.record()
        torch.cuda.synchronize(device)
        ph["bwd"].append(b0.elapsed_time(b1))

        a0 = torch.cuda.Event(enable_timing=True)
        a1 = torch.cuda.Event(enable_timing=True)
        a0.record()
        B.masked_adam_step(opt, B._train_cull_mask(None, dyn))
        a1.record()
        torch.cuda.synchronize(device)
        ph["adam"].append(a0.elapsed_time(a1))

        # checkpoint: fresh eval mask + union stats + stale-vs-fresh audit
        if (it + 1) % 50 == 49:  # non-densify steps only (densify would mark_dirty)
            with torch.no_grad():
                eval_forward_fn(params, eval_idx, sampling_ratio=1.0)
            fresh_em = cull_slot("eval")
            stats = union_stats("step%d" % (it + 1), eval_mask_override=fresh_em)
            print("UNION " + json.dumps(stats), flush=True)
            tm = cull_slot("train")
            if stale_eval is not None and tm is not None and stale_eval.numel() == tm.numel():
                prune_stale = tm & ~stale_eval
                wrongly = prune_stale & fresh_em
                print("AUDIT " + json.dumps({
                    "step": it + 1,
                    "stale_age_steps": 50,
                    "n_prune_stale": int(prune_stale.sum().item()),
                    "n_wrongly_visible": int(wrongly.sum().item()),
                    "wrong_rate": float(wrongly.float().mean().item()) if prune_stale.sum().item() else 0.0,
                    "wrong_of_stale": float(wrongly.sum().item() / max(1, int(prune_stale.sum().item()))),
                }), flush=True)
            else:
                stale_n = stale_eval.numel() if stale_eval is not None else -1
                now_n = tm.numel() if tm is not None else -1
                print("AUDIT_SKIP stale=" + str(stale_n) + " now=" + str(now_n), flush=True)
            stale_eval = fresh_em.detach().clone()

        if is_densify_step and it < 1500:
            d0 = torch.cuda.Event(enable_timing=True)
            d1 = torch.cuda.Event(enable_timing=True)
            d0.record()
            grads = means.grad
            n_old = means.shape[0]
            dup_idx = (
                grads.norm(dim=-1) > DENSIFY_THRESH
            ).nonzero().flatten() if grads is not None else torch.tensor([], device=device)
            old_m, old_q, old_s, old_o, old_c = means, quats, scales, opacities, sh
            new_m, new_q, new_s, new_o, new_c = _densify_gaussians(
                means, quats, scales, opacities, sh, grads, threshold=DENSIFY_THRESH,
            )
            new_m, new_q, new_s, new_o, new_c = _prune_gaussians(
                new_m, new_q, new_s, new_o, new_c, opacity_threshold=PRUNE_THRESH,
            )
            n_new = new_m.shape[0]
            if n_new != n_old:
                pre_map = torch.cat([torch.arange(n_old, device=device), dup_idx])
                keep = (new_o > PRUNE_THRESH).nonzero().flatten()
                old_to_new = pre_map[keep]
                with torch.no_grad():
                    means, quats, scales, opacities, sh = (
                        new_m.detach(), new_q.detach(), new_s.detach(),
                        new_o.detach(), new_c.detach(),
                    )
                for _t in (means, quats, scales, opacities, sh):
                    _t.requires_grad_(True)
                params = (means, quats, scales, opacities, sh)
                sync_optimizer_state_for_topology_change(
                    opt, old_to_new,
                    means=(old_m, means), quats=(old_q, quats),
                    scales=(old_s, scales), opacities=(old_o, opacities),
                    colors=(old_c, sh),
                )
                dyn.mark_dirty()
            d1.record()
            torch.cuda.synchronize(device)
            ph["densify"].append(d0.elapsed_time(d1))

        opt.zero_grad(set_to_none=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_end.record()
        torch.cuda.synchronize(device)
        ph["total"].append(t0.elapsed_time(e_end))

    def avg(x):
        return float(sum(x) / len(x)) if x else 0.0

    def window(lo, hi):
        w = {k: [v for i, v in enumerate(ph[k]) if lo <= i < hi] for k in ph}
        return {k + "_ms": avg(v) for k, v in w.items()}

    print("CULL " + json.dumps(cull_meas), flush=True)
    print("WINDOWS " + json.dumps({
        "early_0_50": window(0, 50),
        "mid_50_100": window(50, 100),
        "late_100_end": window(100, steps),
    }), flush=True)
    print("PHASES " + json.dumps({
        "fwd_ms": avg(ph["fwd"]), "bwd_ms": avg(ph["bwd"]),
        "adam_ms": avg(ph["adam"]), "densify_ms": avg(ph["densify"]),
        "eg_refresh_ms": avg(ph["eg_refresh"]), "total_ms": avg(ph["total"]),
        "n_densify_events": len(ph["densify"]),
        "final_N": int(means.shape[0]),
    }), flush=True)


if __name__ == "__main__":
    main()