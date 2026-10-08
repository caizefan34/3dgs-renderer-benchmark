#!/usr/bin/env python3
"""P11 Gate D1: densification-stat timing decomposition (pre-15K states only).

States: stategen 2000/5000/10000/14000 on room/bicycle/garden.
Measured separately (publication trainer semantics, NOT DefaultStrategy):
  1. raster backward absgrad=False
  2. raster backward absgrad=True
  3. means2d.absgrad clone/read
  4. visibility mask / gather
  5. pixel-space x/y scaling
  6. L2 norm
  7. xyz_gradient_accum / denom updates
  8. max-radii update
  9. total per-step densification-stat update (the trainer's exact block)

Also verifies the exact formula (source-extracted) and computes the D-GATE
joint removable budget.
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

A100_PEAK_BW_GB_S = 1555.0
LAUNCH_OVERHEAD_US = 2.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/root/epic05_p11/dumps/P11_DENSIFICATION_TIMING.json")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(0)

    C.bootstrap_gmain()
    C.import_recipe()
    gpu_rec = C.assert_clean_gpu(args.gpu)
    ssim_fn = C.SepSSIM(device="cuda")

    from dataset import GTDataset
    from gaussian_model import GaussianModel

    # ---- static formula verification (source lines) ----
    gm_src = open(os.path.join(C.REFERENCE_V1_DIR, "gaussian_model.py")).read()
    formula = {
        "abs_component_accumulation_location":
            "raster backward kernel fills v_means2d_abs (atomicAdd of |v|) when "
            "absgrad=True; exposed as means2d.absgrad by rendering.py",
        "verified_add_densification_stats_source": extract_fn(gm_src, "def add_densification_stats"),
        "verified_max_radii2D_trainer_block": (
            "model.max_radii2D[visibility_filter] = torch.max("
            "model.max_radii2D[visibility_filter], "
            "radii[visibility_filter].float().max(dim=-1).values)"),
        "width_height_scaling": "grad[:, 0] *= width / 2.0; grad[:, 1] *= height / 2.0 "
                                "(normalized [-1,1] -> pixel space; matches gsplat "
                                "DefaultStrategy L221-225 semantics)",
        "visible_filter": "(radii > 0).any(dim=-1) on meta['radii'] (trainer L579)",
        "norm": "torch.norm(grad_2d, dim=-1, keepdim=True) on the gathered [N_vis, 2]",
        "persistent_accum_count": "xyz_gradient_accum[vis] += grad_norm; "
                                  "denom[vis] += 1 (persistent [N,1] buffers)",
        "semantics_note": "publication trainer semantics (reference_v1), NOT gsplat "
                          "DefaultStrategy: no n_cameras multiplicative factor, "
                          "threshold applied later as accum/denom >= 0.0008",
    }

    per_scene = {}
    for scene in C.SCENES:
        dataset = GTDataset(scene=scene, repo_root=C.REPO, resolution="1080p",
                            device="cuda", background="black")
        seq = C.camera_sequence_for(scene)
        states = {}
        for it in (2000, 5000, 10000, 14000):
            sp = os.path.join(C.STATEGEN_DIR, scene, f"state_{it}.pt")
            if not os.path.exists(sp):
                print(f"[{scene}] MISSING state {it}; skipped")
                continue
            model, _, meta = C.load_model_for_dataset(scene, dataset, state_path=sp)
            cam_idx = int(seq[it % len(seq)])
            cam, gt = dataset.get_item(cam_idx)
            gt = gt.contiguous()

            # produce the real post-backward artifacts once (absgrad filled)
            r0, image, radii_full, means2d = C.render_with_meta(
                model, cam, model.active_sh_degree)
            L1 = F.l1_loss(image, gt)
            dssim = ssim_fn(image, gt)
            loss = 0.8 * L1 + 0.2 * dssim
            loss.backward()
            radii = radii_full[0]
            visibility_filter = (radii > 0).any(dim=-1)
            n_visible = int(visibility_filter.sum())
            del r0, image, loss, L1, dssim

            N = int(model._xyz.shape[0])
            W_, H_ = int(cam.image_width), int(cam.image_height)
            absgrad = means2d.absgrad            # [1, N, 2]
            grad2 = absgrad[0]                   # [N, 2] (view)
            vis = visibility_filter
            accum = model.xyz_gradient_accum
            denom = model.denom

            res = {
                "state_iteration": it, "N": N, "n_visible": n_visible,
                "camera_index": cam_idx, "image_wh": [W_, H_],
            }

            # ---- 1/2. raster backward absgrad off/on (external adjoint) ----
            # exact external adjoint from the loss path
            r0b, imageb, _, means2db = C.render_with_meta(
                model, cam, model.active_sh_degree)
            lossg = 0.8 * F.l1_loss(imageb, gt) + 0.2 * ssim_fn(imageb, gt)
            v_image = torch.autograd.grad(lossg, r0b)[0].detach().contiguous()
            del r0b, imageb, lossg, means2db

            def bench_raster_bwd(absgrad_flag):
                def setup():
                    rr0, _, _, m2d, inputs = C.render_with_meta(
                        model, cam, model.active_sh_degree, absgrad=absgrad_flag,
                        return_inputs=True)
                    return rr0, inputs
                def timed(ctx):
                    torch.autograd.grad(ctx[0], ctx[1], grad_outputs=v_image)
                # split-bench (setup untimed): reuse Gate L3 helper inline
                for _ in range(args.warmup):
                    timed(setup())
                torch.cuda.synchronize()
                meds = []
                for _ in range(args.reps):
                    evs = [torch.cuda.Event(enable_timing=True)
                           for _ in range(args.iters + 1)]
                    evs[0].record()
                    for i in range(args.iters):
                        ctx = setup()
                        evs[i].record()
                        timed(ctx)
                        evs[i + 1].record()
                    torch.cuda.synchronize()
                    ts = np.array([evs[i].elapsed_time(evs[i + 1])
                                   for i in range(args.iters)])
                    meds.append(float(np.median(ts)))
                return {"rep_medians_ms": meds,
                        "median_of_medians_ms": float(np.median(meds))}

            res["raster_bwd_absgrad_off"] = bench_raster_bwd(False)
            res["raster_bwd_absgrad_on"] = bench_raster_bwd(True)
            res["absgrad_aux_cost_ms"] = (
                res["raster_bwd_absgrad_on"]["median_of_medians_ms"] -
                res["raster_bwd_absgrad_off"]["median_of_medians_ms"])

            # ---- 3. absgrad clone/read ----
            res["absgrad_clone"] = C.bench_events(
                lambda: absgrad.clone(), args.warmup, args.iters, args.reps)

            # ---- 4a. visibility mask ----
            res["visibility_mask"] = C.bench_events(
                lambda: (radii > 0).any(dim=-1), args.warmup, args.iters, args.reps)

            # ---- 4b. gather ----
            res["gather_vis"] = C.bench_events(
                lambda: grad2[vis, :2], args.warmup, args.iters, args.reps)

            # ---- 5. pixel-space scaling (clone + two in-place muls) ----
            pre_clone = grad2.clone()
            def do_scale():
                g = pre_clone.clone()
                g[:, 0] *= W_ / 2.0
                g[:, 1] *= H_ / 2.0
                return g
            res["pixel_space_scaling"] = C.bench_events(
                do_scale, args.warmup, args.iters, args.reps)

            # ---- 6. L2 norm on gathered grads ----
            g2 = grad2.clone()
            g2[:, 0] *= W_ / 2.0
            g2[:, 1] *= H_ / 2.0
            grad_vis = g2[vis, :2].contiguous()
            res["l2_norm"] = C.bench_events(
                lambda: torch.norm(grad_vis, dim=-1, keepdim=True),
                args.warmup, args.iters, args.reps)

            # ---- 7. accum/denom updates ----
            gn = torch.norm(grad_vis, dim=-1, keepdim=True)
            def do_accum():
                accum[vis] += gn
                denom[vis] += 1
            res["accum_denom_update"] = C.bench_events(
                do_accum, args.warmup, args.iters, args.reps)

            # ---- 8. max-radii update (trainer's exact block) ----
            def do_maxradii():
                model.max_radii2D[vis] = torch.max(
                    model.max_radii2D[vis],
                    radii[vis].float().max(dim=-1).values)
            res["max_radii_update"] = C.bench_events(
                do_maxradii, args.warmup, args.iters, args.reps)

            # ---- 9. total per-step densification-stat update ----
            def do_total():
                model.max_radii2D[vis] = torch.max(
                    model.max_radii2D[vis],
                    radii[vis].float().max(dim=-1).values)
                model.add_densification_stats(means2d, vis,
                                              width=W_, height=H_)
            res["total_per_step_stat_update"] = C.bench_events(
                do_total, args.warmup, args.iters, args.reps)

            # ---- T_iter at this state (from L3 dump if present) ----
            titer_ms = None
            l3path = "/root/epic05_p11/dumps/P11_L3_RENDERER_BWD.json"
            if os.path.exists(l3path):
                try:
                    l3 = json.load(open(l3path))
                    t = l3["per_scene"][scene].get(f"pre15k_{it}", {}).get("titer", {})
                    titer_ms = t.get("T_iter_ms")
                except Exception:
                    pass
            res["T_iter_ms"] = titer_ms

            def mom(key):
                return res[key]["median_of_medians_ms"]

            chain_sum = (mom("visibility_mask") + mom("gather_vis") +
                         mom("pixel_space_scaling") + mom("l2_norm") +
                         mom("accum_denom_update"))
            res["measured_chain_sum_excluding_maxradii_ms"] = chain_sum
            res["measured_total_stat_update_ms"] = mom("total_per_step_stat_update")
            states[str(it)] = res
            print(f"[{scene}@{it}] N={N:,} vis={n_visible:,} "
                  f"absgrad_aux={res['absgrad_aux_cost_ms']*1000:.1f}us "
                  f"chain={chain_sum*1000:.1f}us total={mom('total_per_step_stat_update')*1000:.1f}us "
                  f"T_iter={titer_ms}")

        per_scene[scene] = {"states": states}

    # ---- D-GATE computation ----
    dgate = compute_dgate(per_scene)
    record = {
        "phase": "P11 Gate D1 - densification-stat timing decomposition",
        "gpu": gpu_rec,
        "protocol": {"warmup": args.warmup, "measurements": args.iters,
                     "reps": args.reps, "timer": "CUDA events"},
        "formula_verification": formula,
        "per_scene": per_scene,
        "d_gate": dgate,
    }
    C.dump_json(record, args.out)


def extract_fn(src, signature):
    lines = src.splitlines()
    out = []
    started = False
    indent = None
    for i, l in enumerate(lines):
        if l.startswith(signature):
            started = True
            indent = len(l) - len(l.lstrip())
            out.append(l.strip())
            continue
        if started:
            if l.strip() and (len(l) - len(l.lstrip())) <= indent and not l.startswith(" " * (indent + 1)):
                break
            out.append(l.rstrip())
    return "\n".join(out[-60:]).strip()[-3000:]


def compute_dgate(per_scene):
    """Joint removable budget: avoidable absgrad auxiliary traffic in renderer
    backward + post-backward tensor-op chain, as % E2E.

    conservative (traffic/launch floor only):
      - absgrad aux: v_means2d_abs zero-fill (N*8B) + atomic write (N*8B) +
        read-back in the chain (N*8B) at peak BW  [the sink consumes it in-kernel]
      - chain: clone (rd+wr N*8B), gather (rd N*8B wr Nvis*8B), norm interm
        (rd Nvis*8B wr Nvis*4B), mask (N reads), launches (~8 * 2.5us)
      - persistent accum/denom/max_radii updates REMAIN (a sink still performs
        them): only their launch overhead is counted as removable.
    moderate: conservative + (measured absgrad aux cost - its write floor) +
      (measured chain total - estimated unavoidable sink cost of the persistent
      updates at peak BW).
    optimistic: measured absgrad_aux + measured total stat-update chain as-is
      (upper bound; assumes a zero-cost sink epilogue).
    Campaign-level E2E: pre-15K fraction only (iters < 15000 of 30000), so the
    campaign-average removable = pre15k_value * 0.5.
    """
    bw = A100_PEAK_BW_GB_S * 1e9
    rows = []
    for scene, rec in per_scene.items():
        for it, res in rec["states"].items():
            N = res["N"]; Nvis = res["n_visible"]
            titer = res.get("T_iter_ms")
            if not titer:
                continue
            # conservative floor
            absgrad_traffic = 3 * N * 8  # zero-fill + atomic write + chain read
            chain_traffic = (2 * N * 8 + (N + Nvis) * 8 + Nvis * 12 + N)
            launches = 8
            cons_ms = (absgrad_traffic + chain_traffic) / bw * 1000.0 + \
                launches * LAUNCH_OVERHEAD_US / 1000.0
            # moderate
            write_floor_ms = absgrad_traffic / bw * 1000.0
            mod_ms = cons_ms + max(0.0, res["absgrad_aux_cost_ms"] - write_floor_ms) + \
                max(0.0, res["measured_chain_sum_excluding_maxradii_ms"] -
                    ((2 * Nvis * 8 + 2 * N * 4) / bw * 1000.0))
            # optimistic
            opt_ms = res["absgrad_aux_cost_ms"] + res["measured_total_stat_update_ms"]
            rows.append({
                "scene": scene, "iteration": int(it), "N": N, "Nvis": Nvis,
                "T_iter_ms": titer,
                "conservative_ms": cons_ms,
                "moderate_ms": mod_ms,
                "optimistic_ms": opt_ms,
                "conservative_pct_titer": cons_ms / titer * 100,
                "moderate_pct_titer": mod_ms / titer * 100,
                "optimistic_pct_titer": opt_ms / titer * 100,
                # campaign-average: only pre-15K iterations have this work
                "conservative_pct_campaign_e2e": cons_ms / titer * 100 * 0.5,
                "moderate_pct_campaign_e2e": mod_ms / titer * 100 * 0.5,
                "optimistic_pct_campaign_e2e": opt_ms / titer * 100 * 0.5,
            })
    if not rows:
        return {"error": "no states with T_iter available"}
    summary = {
        "n_cells": len(rows),
        "mean_conservative_pct_titer": float(np.mean([r["conservative_pct_titer"] for r in rows])),
        "mean_moderate_pct_titer": float(np.mean([r["moderate_pct_titer"] for r in rows])),
        "mean_optimistic_pct_titer": float(np.mean([r["optimistic_pct_titer"] for r in rows])),
        "mean_conservative_pct_campaign_e2e": float(np.mean([r["conservative_pct_campaign_e2e"] for r in rows])),
        "mean_moderate_pct_campaign_e2e": float(np.mean([r["moderate_pct_campaign_e2e"] for r in rows])),
        "mean_optimistic_pct_campaign_e2e": float(np.mean([r["optimistic_pct_campaign_e2e"] for r in rows])),
        "worst_conservative_pct_campaign_e2e": float(np.max([r["conservative_pct_campaign_e2e"] for r in rows])),
    }
    summary["gate"] = ("ADVANCE" if summary["mean_conservative_pct_campaign_e2e"] >= 2.0
                       else "DROP")
    summary["gate_rule"] = "conservative campaign-average joint removable >= 2.0% E2E"
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()
