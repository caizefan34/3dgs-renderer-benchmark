#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/epic05_p17")
import p17_p0_oracle as P

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=("room", "bicycle"))
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--out", default="/root/epic05_p17")
    args = ap.parse_args()

    units_dir = os.path.join(args.out, "P17_UNITS", args.scene)
    Path(units_dir).mkdir(parents=True, exist_ok=True)

    camset = json.load(open(os.path.join(args.out, "P17_CAMERA_SET.json")))
    assert camset["scene"] == args.scene
    probes, audits = camset["probe_ids"], camset["audit_ids"]
    state = json.load(open(os.path.join(args.out, "P17_STATE_IDENTITY.json")))
    assert state["scene"] == args.scene

    P.verify_identity(args.out, args.scene)
    ctx = P.SceneCtx(args.scene)
    units = []
    for ci in probes:
        ctx.cur_ci = ci
        cam, gt = ctx.cam_gt(ci)
        H0, W0 = int(cam.image_height), int(cam.image_width)
        K0 = cam.K.detach().cpu().numpy().astype(np.float64)
        K0t = torch.as_tensor(K0, device="cuda", dtype=torch.float32)
        camK = {d: torch.as_tensor(P.scale_K(K0, d), device="cuda",
                                   dtype=torch.float32) for d in P.DS}

        # capture the full-res raw grads right after the reference backward
        _, ref_flat, ref_frame, ref_loss, ref_meta = P.render_backward_grads(
            ctx, cam, K0t, W0, H0, P.EPS2D_FULL)
        ref_pg = ctx.grad_groups()
        _, orb_flat, orb_frame, orb_loss, orb_meta = P.render_backward_grads(
            ctx, cam, K0t, W0, H0, P.EPS2D_FULL)
        rgb_max = float((ref_frame - orb_frame).abs().max())
        loss_rel = abs(ref_loss - orb_loss) / (abs(ref_loss) + 1e-30)
        grad_cos = {}
        all_finite = True
        for key in ("means", "scales", "quats", "opacity", "sh0", "shN"):
            if not (torch.isfinite(ref_flat[key]).all()
                    and torch.isfinite(orb_flat[key]).all()):
                all_finite = False
                continue
            grad_cos[key] = P.cosine(ref_flat[key], orb_flat[key])
        vis1 = ctx.visible_ids(cam, K0, W0, H0, P.EPS2D_FULL)
        vis1b = ctx.visible_ids(cam, K0, W0, H0, P.EPS2D_FULL)
        vis_eq = bool(np.array_equal(np.sort(vis1), np.sort(vis1b)))
        parity_pass = (rgb_max <= 1e-6 and loss_rel <= 1e-7 and all_finite
                       and vis_eq and all(
                           v is not None and v >= 0.999999
                           for v in grad_cos.values()))
        if not parity_pass:
            P.jwrite(os.path.join(units_dir, f"camera_{ci}.json"), {
                "scene": args.scene, "camera_id": ci, "tau": 2.0,
                "d1_parity_pass": False,
                "oracle_interaction_ratio": None,
                "full_cost_fraction_d2plus": None,
                "full_cost_fraction_d4plus": None,
                "full_cost_fraction_d8": None,
                "metrics": {k: {"cosine": None, "rel_l2": None,
                                "top10_sign": None}
                            for k in ("means", "scales", "quats", "opacity",
                                      "sh0", "shN")},
                "step": {"reference_improving": None, "R_step": None,
                         "P_audit": None}})
            units.append({"camera_id": ci, "parity_pass": False})
            continue

        coarse, coarse_vis = {}, {}
        gt_np = gt.cpu().numpy()
        for d in P.DS:
            Hd, Wd = H0 // d, W0 // d
            eps_d = P.EPS2D_FULL / (d * d)
            gt_coarse = torch.as_tensor(
                P.area_avg_2d(gt_np, d), device="cuda", dtype=torch.float32)
            _, g_d, frame_d, loss_d, meta_d = P.render_backward_grads(
                ctx, cam, camK[d], Wd, Hd, eps_d, gt_coarse=gt_coarse)
            vis_d = ctx.visible_ids(cam, P.scale_K(K0, d), Wd, Hd, eps_d)
            g_d_pg = {k: v.reshape(ctx.N, -1) for k, v in g_d.items()}
            coarse[d] = {"grads": g_d_pg, "flat": g_d, "loss": loss_d}
            coarse_vis[d] = vis_d

        counts1, n_isects1 = ctx.intersect_counts(
            cam, K0, W0, H0, P.EPS2D_FULL, vis1)
        rad1, _, _, conics1 = ctx.producer_full(
            cam, K0, W0, H0, P.EPS2D_FULL)
        vis_rows = rad1[:, 0] > 0
        sig_min1, sig_max1 = P.sigma_min_from_conics(conics1)
        sig_min1[~vis_rows] = 0.0
        sig_max1[~vis_rows] = 0.0
        coarse_counts = {}
        for d in P.DS:
            coarse_counts[d], _ = ctx.intersect_counts(
                cam, P.scale_K(K0, d), W0 // d, H0 // d,
                P.EPS2D_FULL / (d * d), coarse_vis[d])

        N = ctx.N
        d_full = P.assign_d(sig_min1.cpu(), args.tau)
        # full-res raw grads captured right after the reference backward
        ref_grads = ref_pg
        hyb = {k: torch.zeros_like(ref_grads[k]) for k in ref_grads}
        assigned = torch.zeros(N, dtype=torch.int64)
        cost_by_d = torch.zeros(N, dtype=torch.float64)
        vis1_t = torch.as_tensor(vis1, dtype=torch.int64)
        cnt1_full = torch.zeros(N, dtype=torch.float64)
        cnt1_full[vis1_t] = torch.as_tensor(counts1, dtype=torch.float64)
        for d in P.DS:
            sel = (d_full == d)
            if not bool(sel.any()):
                continue
            vis_d = torch.as_tensor(coarse_vis[d], dtype=torch.int64)
            cnt_d_full = torch.zeros(N, dtype=torch.float64)
            cnt_d_full[vis_d] = torch.as_tensor(
                coarse_counts[d], dtype=torch.float64)
            for k in ref_grads:
                if k in ("sh0", "shN"):
                    continue  # sh0/shN are views of the shs block
                g_d_full = torch.zeros_like(ref_grads[k])
                g_d_full[vis_d] = coarse[d]["grads"][k][: vis_d.numel()]
                hyb[k] = torch.where(sel.unsqueeze(1), g_d_full, hyb[k])
            assigned[sel] = d
            cost_by_d[sel] = cnt_d_full[sel]
        cnt1_tot = float(cnt1_full.sum())
        C_oracle = (float(cost_by_d.sum()) / cnt1_tot
                    if cnt1_tot > 0 else None)
        fr = {}
        for dd, lab in ((2, "d2plus"), (4, "d4plus"), (8, "d8")):
            mask = assigned >= dd
            fr[lab] = (float(cnt1_full[mask].sum()) / cnt1_tot
                       if cnt1_tot > 0 else None)
        pop_full = torch.zeros(N, dtype=torch.bool)
        pop_full[torch.where(assigned > 1)[0]] = True
        n_d2 = int(pop_full.sum())
        mtr = (P.metrics_for(ctx, pop_full, ref_grads, hyb) if n_d2 > 0
               else {k: {"cosine": None, "rel_l2": None, "top10_sign": None}
                     for k in ("means", "scales", "quats", "opacity",
                               "sh0", "shN")})

        hyb_flat = {k: v.reshape(-1) for k, v in hyb.items()}
        L_before_d = P.dense_adam_step(ctx, cam, K0, W0, H0)
        L_after_dense = P.poststep_eval(ctx, cam, K0, W0, H0, gt)
        L_before_h = P.apply_hybrid_adam_step(ctx, cam, K0, W0, H0, hyb_flat)
        L_after_hyb = P.poststep_eval(ctx, cam, K0, W0, H0, gt)
        assert abs(L_before_d - L_before_h) < 1e-12, "L_before mismatch"
        G_dense = L_before_d - L_after_dense
        G_hyb = L_before_h - L_after_hyb
        # prereg section 11: G_dense <= 0 => STEP_NONIMPROVING_REFERENCE
        ref_improving = bool(G_dense > 0.0)
        R_step = (G_hyb / max(G_dense, 1e-12)) if ref_improving else None
        ctx.restore_initial()
        L_audit_before = P.audit_losses(ctx, audits)
        P.dense_adam_step(ctx, cam, K0, W0, H0)
        L_audit_after_dense = P.audit_losses(ctx, audits)
        P.apply_hybrid_adam_step(ctx, cam, K0, W0, H0, hyb_flat)
        L_audit_after_hyb = P.audit_losses(ctx, audits)
        P_audit_vals = [
            (L_audit_after_hyb[i] - L_audit_after_dense[i])
            / (L_audit_before[i] + 1e-30) for i in range(len(audits))]
        P_audit = float(np.mean(P_audit_vals))

        unit = {
            "scene": args.scene,
            "camera_id": ci,
            "tau": args.tau,
            "d1_parity_pass": True,
            "oracle_interaction_ratio": C_oracle,
            "full_cost_fraction_d2plus": fr["d2plus"],
            "full_cost_fraction_d4plus": fr["d4plus"],
            "full_cost_fraction_d8": fr["d8"],
            "metrics": {k: {kk: v[kk] for kk in ("cosine", "rel_l2",
                                                 "top10_sign")}
                        for k, v in mtr.items()},
            "step": {"reference_improving": ref_improving,
                     "R_step": R_step, "P_audit": P_audit},
            "detail": {
                "n_visible_full": int(vis1.shape[0]),
                "n_isects_full": n_isects1,
                "n_assigned_d2plus": n_d2,
                "cost_by_d": {"d1": int((d_full == 1).sum().item()),
                              "d2": int((d_full == 2).sum().item()),
                              "d4": int((d_full == 4).sum().item()),
                              "d8": int((d_full == 8).sum().item())},
                "L_before": L_before_d,
                "L_after_dense_probe": L_after_dense,
                "L_after_hyb_probe": L_after_hyb,
                "G_dense": G_dense, "G_hyb": G_hyb,
                "P_audit_per_cam": P_audit_vals,
                "L_audit_before": L_audit_before,
                "L_audit_after_dense": L_audit_after_dense,
                "L_audit_after_hyb": L_audit_after_hyb,
            },
        }
        P.jwrite(os.path.join(units_dir, f"camera_{ci}.json"), unit)
        units.append({"camera_id": ci, "parity_pass": True,
                      "C_oracle": C_oracle, "n_d2": n_d2,
                      "R_step": R_step, "P_audit": P_audit})
        print(f"[tau{args.tau}] {args.scene} cam{ci}: "
              f"C={C_oracle:.4f} n_d2+={n_d2} R={R_step} P={P_audit:.2e}",
              flush=True)
    print(f"TAU{args.tau}_DONE {args.scene} units={len(units)}", flush=True)


if __name__ == "__main__":
    main()
