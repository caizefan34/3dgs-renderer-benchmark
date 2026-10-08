#!/usr/bin/env python3
"""H0 rollout harness (H0_PREREG.md sections 3-9).

Topology-frozen K-step rollouts from a FROZEN C0 checkpoint for arms
NATIVE / E2D / OPA, evaluating the EXACT native renderer after every step
(measurement only, excluded from step timing). Emits per-rollout trajectories
and the section-8 progress/efficiency/crossing metrics.

Decision-independent core engine. Runs on whatever checkpoints exist; the full
6-cell scientific grid additionally needs room@5K + bicycle@5K
(H0_CHECKPOINT_BLOCKER). A smoke test on an existing cell validates mechanics
ONLY and is NOT the scientific result.

Reuses frozen primitives from n1_core.py: bootstrap_c0, build_model,
StateMaster, render_view, make_fused_loss, load_scene_dataset,
load_checkpoint, param_update_norm. Renderer: rasterize_gaussian_higs_dynamic
(c0 branch, verbatim from the frozen trainer).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "/root/epic05_n1")
import numpy as np
import torch
import torch.nn.functional as F

import n1_core as C

EPS2D_NATIVE = C.EPS2D      # 0.3
OPA_EPS = 1e-7
LAMBDA = C.LAMBDA_DSSIM     # 0.2

# Frozen continuation schedules (prereg sec.5 E2D, sec.6 OPA)
SCHED = {
    "E2D": {4: [2.00, 1.60, 1.30, 1.00],
            8: [2.00, 1.85, 1.70, 1.55, 1.40, 1.25, 1.10, 1.00]},
    "OPA": {4: [0.60, 0.75, 0.90, 1.00],
            8: [0.60, 0.68, 0.76, 0.84, 0.90, 0.94, 0.97, 1.00]},
}


# ---- SepSSIM (verbatim from publication_trainer_p12_n1snap.py) --------------
class SepSSIM:
    def __init__(self, device="cuda", window_size=11, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + 0.01 ** 2) * (2 * spt + 0.03 ** 2) / \
                   ((mu_p2 + mu_t2 + 0.01 ** 2) * (sp2 + st2 + 0.03 ** 2))
        return 1.0 - ssim_map.mean()


def native_objective(img_hwc, gt_hwc, ssim_fn):
    """L = 0.8*L1 + 0.2*(1-SSIM), exact c0 training-loss formula on HWC."""
    L1 = F.l1_loss(img_hwc, gt_hwc)
    dssim = ssim_fn(img_hwc, gt_hwc)
    return (1.0 - LAMBDA) * L1 + LAMBDA * dssim


def opa_opacity(o, gamma):
    o_c = torch.clamp(o, 0.0, 1.0 - OPA_EPS)
    if gamma == 1.0:
        return o_c
    return 1.0 - (1.0 - o_c) ** gamma


def render_step(model, cam, sh_degree, arm, K, j):
    """TRAIN render for 1-based step j under the arm's continuation parameter."""
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    if arm == "E2D":
        eps2d = SCHED["E2D"][K][j - 1] * EPS2D_NATIVE
        opac = model.get_opacity
    elif arm == "OPA":
        eps2d = EPS2D_NATIVE
        opac = opa_opacity(model.get_opacity, SCHED["OPA"][K][j - 1])
    else:  # NATIVE
        eps2d = EPS2D_NATIVE
        opac = model.get_opacity
    out = rasterize_gaussian_higs_dynamic(
        model.get_xyz, model.get_rotation, model.get_scaling, opac, model.get_features,
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=eps2d)
    info = out["densification_info"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
    return frame.clamp(0, 1), info["radii"], info["means2d"]


def native_metrics(model, design_cams, hold_cams, ds, ssim_fn):
    """EXACT native renderer over the 8 frozen cams; design + held-out objective, PSNR, SSIM."""
    def one(cam_idx):
        cam, gt = ds.get_item(cam_idx)
        with torch.no_grad():
            img, _, _ = C.render_view(model, cam, model.active_sh_degree)
        obj = float(native_objective(img, gt, ssim_fn).item())
        b = img.unsqueeze(0).permute(0, 3, 1, 2)
        g = gt.unsqueeze(0).permute(0, 3, 1, 2)
        mse = F.mse_loss(b, g)
        psnr = float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0
        ssim = float(1.0 - ssim_fn(img, gt).item())
        return obj, psnr, ssim
    d = [one(c) for c in design_cams]
    h = [one(c) for c in hold_cams]
    return {
        "design_obj": float(np.mean([x[0] for x in d])),
        "holdout_obj": float(np.mean([x[0] for x in h])),
        "design_psnr": float(np.mean([x[1] for x in d])),
        "holdout_psnr": float(np.mean([x[1] for x in h])),
        "design_ssim": float(np.mean([x[2] for x in d])),
        "holdout_ssim": float(np.mean([x[2] for x in h])),
    }


def run_rollout(model, master, ds, ssim_fn, arm, K, fold_name, update_pool, heldout, offset, time_steps):
    master.restore()
    seq = [update_pool[(offset + i) % 4] for i in range(K)]  # fixed cyclic offset
    ev0 = native_metrics(model, update_pool, heldout, ds, ssim_fn)
    traj = [{"step": 0, "native": ev0, "update_norm": master.param_update_norm()}]
    step_times = []
    for j in range(1, K + 1):
        cam, gt = ds.get_item(seq[j - 1])
        model.optimizer.zero_grad(set_to_none=True)
        if time_steps:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        img, _, _ = render_step(model, cam, model.active_sh_degree, arm, K, j)
        loss = native_objective(img, gt, ssim_fn)
        loss.backward()
        model.optimizer.step()          # topology frozen: NO densify/prune/reset
        if time_steps:
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t0) * 1000.0)
        ev = native_metrics(model, update_pool, heldout, ds, ssim_fn)
        traj.append({"step": j, "loss": float(loss.item()), "native": ev,
                     "update_norm": master.param_update_norm(),
                     "nan_inf": int(not math.isfinite(loss.item()))})
    return {"arm": arm, "K": K, "fold": fold_name, "offset": offset,
            "update_seq": seq, "trajectory": traj,
            "step_times_ms": step_times if time_steps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--arms", default="NATIVE,E2D,OPA")
    ap.add_argument("--Ks", default="4,8")
    ap.add_argument("--folds", default="A,B")
    ap.add_argument("--offsets", default="0,1,2,3")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ident = C.bootstrap_c0()
    fused = C.make_fused_loss()
    ssim_fn = SepSSIM(device="cuda")
    ds = C.load_scene_dataset(args.scene)

    # frozen 8-camera MOONSHOT set + folds (prereg sec.3)
    n = len(ds)
    ids = [int(round(n * (i + 0.5) / 8)) for i in range(8)]
    fold_defs = {
        "A": {"update_pool": ids[0:4], "heldout": ids[4:8]},
        "B": {"update_pool": ids[4:8], "heldout": ids[0:4]},
    }
    arms = args.arms.split(",")
    Ks = [int(x) for x in args.Ks.split(",")]
    folds = args.folds.split(",")
    offsets = [int(x) for x in args.offsets.split(",")]

    ckpt = C.load_checkpoint(args.scene, args.iter)
    model = C.build_model(ckpt)
    C.scene_reset()
    master = C.StateMaster(model)

    rollouts = []
    for K in Ks:
        for fold_name in folds:
            fd = fold_defs[fold_name]
            for off in offsets:
                for arm in arms:
                    r = run_rollout(model, master, ds, ssim_fn, arm, K, fold_name,
                                    fd["update_pool"], fd["heldout"], off, args.time)
                    r["scene"] = args.scene
                    r["iter"] = args.iter
                    rollouts.append(r)
                    print(f"  [{args.scene}@{args.iter}] K{K} fold{fold_name} off{off} {arm}: "
                          f"L0_hold={r['trajectory'][0]['native']['holdout_obj']:.6f} "
                          f"LK_hold={r['trajectory'][-1]['native']['holdout_obj']:.6f}", flush=True)

    out = {"audit_type": "H0_ROLLOUTS", "scene": args.scene, "iter": args.iter,
           "arms": arms, "Ks": Ks, "camera_ids_c0_c7": ids, "rollouts": rollouts}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} ({len(rollouts)} rollouts)")


if __name__ == "__main__":
    main()
