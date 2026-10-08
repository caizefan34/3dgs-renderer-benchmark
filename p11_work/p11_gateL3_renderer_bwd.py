#!/usr/bin/env python3
"""P11 Gate L3: renderer backward with externally supplied image adjoint.

Separates loss backward from renderer backward. For each frozen state/camera:
  1. render normally;
  2. generate the exact dL/dimage using the loss path (includes clamp backward);
  3. benchmark renderer backward driven by the precomputed detached image adjoint;
  4. benchmark the full loss.backward() path.

Graphs are rebuilt per timed sample (never reused). Additive closure
(renderer_bwd_external + loss_to_image_bwd ≈ full_backward) is checked.

Also measures T_iter (trainer-faithful full iteration, 6-event phase timing)
at every state for E2E normalization.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C


def bench_split(setup, timed, warmup=20, iters=100, reps=5):
    """CUDA-event timing of `timed(ctx)` only; `setup()` rebuilds context per
    sample (untimed). Events bracket just the timed region on the stream."""
    for _ in range(warmup):
        ctx = setup()
        timed(ctx)
    torch.cuda.synchronize()
    rep_meds = []
    for _ in range(reps):
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
        evs[0].record()
        for i in range(iters):
            ctx = setup()
            evs[i].record()
            timed(ctx)
            evs[i + 1].record()
        torch.cuda.synchronize()
        ts = np.array([evs[i].elapsed_time(evs[i + 1]) for i in range(iters)])
        rep_meds.append(float(np.median(ts)))
    return {
        "protocol": {"warmup": warmup, "measurements": iters, "reps": reps,
                     "timer": "cuda_events", "graph": "rebuilt per sample"},
        "rep_medians_ms": rep_meds,
        "median_of_medians_ms": float(np.median(rep_meds)),
        "mean_of_medians_ms": float(np.mean(rep_meds)),
    }


def make_loss(image, gt, ssim_fn):
    L1 = F.l1_loss(image, gt)
    dssim = ssim_fn(image, gt)
    return (1.0 - C.LAMBDA_DSSIM) * L1 + C.LAMBDA_DSSIM * dssim


def measure_state(scene, state_path, cam_indices, dataset, ssim_fn, args, label):
    model, _, meta = C.load_model_for_dataset(scene, dataset, state_path=state_path)
    out = {"state": {"source": state_path, "sha256": meta["state_sha256"],
                     "iteration": meta["state_iteration"], "N": meta["state_N"],
                     "label": label}}
    seq = C.camera_sequence_for(scene)

    per_cam = []
    for cam_idx in cam_indices:
        cam, gt = dataset.get_item(cam_idx)
        gt = gt.contiguous()

        # ---- precompute the exact external dL/dimage (fresh graph) ----
        r0, image, radii_full, means2d = C.render_with_meta(
            model, cam, model.active_sh_degree)
        loss = make_loss(image, gt, ssim_fn)
        v_image = torch.autograd.grad(loss, r0)[0].detach().contiguous()
        del r0, image, loss, means2d

        # ---- 1. renderer forward (fresh graph per call; the graph build is part
        #         of the trainer's forward phase cost, so it stays in the timed
        #         region; the returned graph is discarded un-consumed) ----
        def do_fwd():
            r0, image, radii_full, means2d = C.render_with_meta(
                model, cam, model.active_sh_degree)
            return r0
        res_fwd = C.bench_events(do_fwd, args.warmup, args.iters, args.reps)

        # ---- 2. renderer backward given external dL/dimage ----
        def setup_bwd():
            r0, image, radii_full, means2d, inputs = C.render_with_meta(
                model, cam, model.active_sh_degree, return_inputs=True)
            # NOTE: means2d is a sibling OUTPUT of the raster op, not an
            # ancestor of r0, so it cannot appear in the inputs list; its
            # absgrad is populated by the same backward kernel, so the cost
            # remains inside this measurement. The five parameter tensors are
            # the differentiable inputs of the renderer.
            return (r0, inputs)
        res_rbwd = bench_split(
            setup_bwd,
            lambda ctx: torch.autograd.grad(ctx[0], ctx[1], grad_outputs=v_image),
            args.warmup, args.iters, args.reps)

        # ---- 3. loss-to-image backward (clamp included; retained graph) ----
        # precompute one detached unclamped render for the loss graph
        with torch.no_grad():
            r0u, _, _, _ = C.render_with_meta(model, cam, model.active_sh_degree)
        r0_leaf = r0u.detach().requires_grad_(True)
        loss_g = make_loss(r0_leaf.clamp(0, 1), gt, ssim_fn)
        res_lbwd = C.bench_events(
            lambda: torch.autograd.grad(loss_g, r0_leaf, retain_graph=True),
            args.warmup, args.iters, args.reps)
        del loss_g, r0_leaf, r0u

        # ---- 5. full loss.backward() (graph rebuilt per sample) ----
        def setup_full():
            model.optimizer.zero_grad(set_to_none=True)
            r0, image, radii_full, means2d = C.render_with_meta(
                model, cam, model.active_sh_degree)
            return make_loss(image, gt, ssim_fn)
        res_full = bench_split(
            setup_full,
            lambda loss: loss.backward(),
            args.warmup, args.iters, args.reps)

        rb = res_rbwd["median_of_medians_ms"]
        lb = res_lbwd["median_of_medians_ms"]
        fb = res_full["median_of_medians_ms"]
        rf = res_fwd["median_of_medians_ms"]
        closure = {
            "renderer_bwd_external_ms": rb,
            "loss_to_image_bwd_ms": lb,
            "sum_ms": rb + lb,
            "full_backward_ms": fb,
            "abs_gap_ms": abs((rb + lb) - fb),
            "rel_gap": abs((rb + lb) - fb) / fb if fb > 0 else None,
        }
        per_cam.append({
            "camera_index": cam_idx,
            "renderer_forward_ms": rf,
            "renderer_backward_external_adjoint": res_rbwd,
            "loss_to_image_backward": res_lbwd,
            "full_backward": res_full,
            "closure": closure,
        })
        print(f"[{scene}/{label} cam{cam_idx}] fwd={rf:.3f} rbwd={rb:.3f} "
              f"lbwd={lb:.3f} full={fb:.3f} closure_gap={closure['rel_gap']*100:.2f}%")

    out["per_camera"] = per_cam
    out["state_summary"] = {
        "renderer_forward_ms": float(np.mean([c["renderer_forward_ms"] for c in per_cam])),
        "renderer_backward_external_ms": float(np.mean(
            [c["renderer_backward_external_adjoint"]["median_of_medians_ms"] for c in per_cam])),
        "loss_to_image_bwd_ms": float(np.mean(
            [c["loss_to_image_backward"]["median_of_medians_ms"] for c in per_cam])),
        "full_backward_ms": float(np.mean(
            [c["full_backward"]["median_of_medians_ms"] for c in per_cam])),
        "mean_abs_closure_gap_pct": float(np.mean(
            [c["closure"]["rel_gap"] * 100 for c in per_cam])),
    }

    # ---- T_iter: trainer-faithful full iteration with 6-event phase timing ----
    st_iter = meta["state_iteration"] or 0
    pre15k = st_iter < 15000
    pev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    phase_ms = {k: [] for k in ("forward", "loss", "backward", "densify", "optimizer")}

    def one_iter(it):
        model.update_learning_rate(it)
        cam_idx = int(seq[(it - 1) % len(seq)])
        cam, gt = dataset.get_item(cam_idx)
        pev[0].record()
        r0, image, radii_full, means2d = C.render_with_meta(
            model, cam, model.active_sh_degree)
        pev[1].record()
        L1 = F.l1_loss(image, gt)
        dssim = ssim_fn(image, gt)
        loss = (1.0 - C.LAMBDA_DSSIM) * L1 + C.LAMBDA_DSSIM * dssim
        pev[2].record()
        loss.backward()
        pev[3].record()
        radii = radii_full[0]
        visibility_filter = (radii > 0).any(dim=-1)
        n_visible = int(visibility_filter.sum())  # trainer does this every iter (D2H)
        if pre15k:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values)
            model.add_densification_stats(means2d, visibility_filter,
                                          width=cam.image_width, height=cam.image_height)
        pev[4].record()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        pev[5].record()
        torch.cuda.synchronize()
        phase_ms["forward"].append(pev[0].elapsed_time(pev[1]))
        phase_ms["loss"].append(pev[1].elapsed_time(pev[2]))
        phase_ms["backward"].append(pev[2].elapsed_time(pev[3]))
        phase_ms["densify"].append(pev[3].elapsed_time(pev[4]))
        phase_ms["optimizer"].append(pev[4].elapsed_time(pev[5]))

    # warmup + measurement (skip densification-event iterations for a clean
    # per-step picture; the stat-update chain is what we time, events at %100
    # do topology changes and are not part of the per-step D-gate budget)
    base = st_iter if st_iter > 0 else 1
    it = base
    warm = 0
    while warm < args.titer_warmup:
        it += 1
        if it % 100 == 0 or (pre15k and it % 3000 == 0):
            continue
        one_iter(it)
        warm += 1
    n_meas = 0
    while n_meas < args.titer_iters:
        it += 1
        if it % 100 == 0 or (pre15k and it % 3000 == 0):
            continue
        one_iter(it)
        n_meas += 1

    titer = {k: float(np.mean(v)) for k, v in phase_ms.items()}
    titer["T_iter_ms"] = float(sum(titer.values()))
    out["titer"] = titer
    print(f"[{scene}/{label}] T_iter={titer['T_iter_ms']:.2f}ms "
          f"(fwd {titer['forward']:.2f} loss {titer['loss']:.2f} "
          f"bwd {titer['backward']:.2f} densify {titer['densify']:.2f} "
          f"opt {titer['optimizer']:.2f}) pre15k={pre15k}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/root/epic05_p11/dumps/P11_L3_RENDERER_BWD.json")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--titer-warmup", type=int, default=10)
    ap.add_argument("--titer-iters", type=int, default=60)
    ap.add_argument("--late-cams", type=int, default=3)
    ap.add_argument("--skip-stategen", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(0)

    C.bootstrap_gmain()
    C.import_recipe()
    gpu_rec = C.assert_clean_gpu(args.gpu)
    ssim_fn = C.SepSSIM(device="cuda")

    from dataset import GTDataset
    datasets = {s: GTDataset(scene=s, repo_root=C.REPO, resolution="1080p",
                             device="cuda", background="black") for s in C.SCENES}

    results = {}
    for scene in C.SCENES:
        dataset = datasets[scene]
        seq = C.camera_sequence_for(scene)
        scene_res = {}

        # late 30K frozen state, 3 cameras
        ckpt = os.path.join(C.FULL30K, f"gmain03_s42_{scene}", "ckpt_30000.pt")
        cams = [int(x) for x in seq[:args.late_cams]]
        scene_res["late_30k"] = measure_state(scene, ckpt, cams, dataset,
                                              ssim_fn, args, "late_30k")

        # pre-15K stategen states, 1 camera each
        if not args.skip_stategen:
            for it in (2000, 5000, 10000, 14000):
                sp = os.path.join(C.STATEGEN_DIR, scene, f"state_{it}.pt")
                if not os.path.exists(sp):
                    print(f"[{scene}] MISSING {sp}; skipping")
                    continue
                cam_idx = int(seq[it % len(seq)])
                scene_res[f"pre15k_{it}"] = measure_state(
                    scene, sp, [cam_idx], dataset, ssim_fn, args, f"pre15k_{it}")

        results[scene] = scene_res

    record = {
        "phase": "P11 Gate L3 - renderer backward with external image adjoint",
        "gpu": gpu_rec,
        "protocol": {"warmup": args.warmup, "measurements": args.iters,
                     "reps": args.reps,
                     "graph_policy": "fresh graph per timed sample; external "
                                     "adjoint precomputed once per (state,camera) "
                                     "and detached",
                     "adjoint_definition": "dL/dimage at the UNCLAMPED renderer "
                                           "output r0 (includes clamp backward); "
                                           "grad_outputs=v_image"},
        "per_scene": results,
    }
    C.dump_json(record, args.out)


if __name__ == "__main__":
    main()
