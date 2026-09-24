#!/usr/bin/env python3
"""
C42 Phase E-1: Gradient Redundancy Profiling.

Measures D-SSIM vs L1 gradient contributions to Gaussian parameters
across training checkpoints (iter 5000 → 30000).

For each checkpoint and each camera:
  1. L1-only backward  → grad norms per param group
  2. D-SSIM-only backward → grad norms per param group
  3. Combined backward → grad norms per param group

Key metric: does ||grad_DSSIM|| / ||grad_total|| decrease over training?
"""
import json, math, sys, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset
from loss import d_ssim_loss

DEVICE = "cuda"
TILE_SIZE = 16
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0
LAMBDA_DSSIM = 0.2

# Checkpoints from the fully-trained v2_16 series
CKPT_BASE = "results/epic05/phase7/phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter{}.pt"
ITERATIONS = [5000, 10000, 15000, 20000, 25000, 30000]
# Also include mid_16 series for early training
CKPT_MID = "results/epic05/phase7/phase7_room_mid_16/phase7_room_mid_16_iter{}.pt"
MID_ITERATIONS = [1000, 2000, 3000]

CAM_INDICES = [0, 50, 100, 150, 200, 250, 300]
PARAM_NAMES = ["xyz", "opacity", "scales", "rotations", "shs"]


def compute_grad_norms(model):
    """Extract L2 gradient norms for each parameter group."""
    norms = {}
    for name in PARAM_NAMES:
        param = getattr(model, name)
        if param.grad is not None:
            norms[name] = float(param.grad.norm().item())
            norms[f"{name}_per_elem"] = float(
                (param.grad.norm() / math.sqrt(param.grad.numel())).item()
            )
        else:
            norms[name] = 0.0
            norms[f"{name}_per_elem"] = 0.0
    return norms


def zero_grads(model):
    for name in PARAM_NAMES:
        param = getattr(model, name)
        if param.grad is not None:
            param.grad = None


def render_with_grad(model, cam, img_w, img_h):
    """Render with gradient tracking enabled. Returns rendered image."""
    data = model.forward()
    rendered, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=img_w, height=img_h,
        tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
    )
    return rendered[0].clamp(0, 1)


def profile_checkpoint(ckpt_path, dataset, cam_indices):
    """Profile gradient norms at one checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=DEVICE)
    N = model.xyz.shape[0]

    results = []
    for cam_idx in cam_indices:
        cam = dataset.get_camera(cam_idx)
        gt_img = dataset.get_gt_image(cam_idx)
        img_w, img_h = cam.image_width, cam.image_height

        # ── L1 only (separate render) ──
        zero_grads(model)
        pred = render_with_grad(model, cam, img_w, img_h)
        l1_val = F.l1_loss(pred, gt_img)
        l1_val.backward()
        l1_grads = compute_grad_norms(model)

        # ── D-SSIM only (separate render) ──
        zero_grads(model)
        pred = render_with_grad(model, cam, img_w, img_h)
        dsim_val = d_ssim_loss(pred, gt_img)
        dsim_val.backward()
        dsim_grads = compute_grad_norms(model)

        # ── Combined (separate render) ──
        zero_grads(model)
        pred = render_with_grad(model, cam, img_w, img_h)
        l1_c = F.l1_loss(pred, gt_img)
        dsim_c = d_ssim_loss(pred, gt_img)
        combined = (1.0 - LAMBDA_DSSIM) * l1_c + LAMBDA_DSSIM * dsim_c
        combined.backward()
        combined_grads = compute_grad_norms(model)

        zero_grads(model)

        # ── PSNR (clean re-render) ──
        with torch.no_grad():
            pred_psnr = render_with_grad(model, cam, img_w, img_h).detach()
            mse = float(((pred_psnr - gt_img) ** 2).mean())
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        results.append({
            "cam_idx": cam_idx,
            "psnr": psnr,
            "l1_loss": float(l1_val.item()),
            "dssim_loss": float(dsim_val.item()),
            "combined_loss": float(combined.item()),
            "loss_ratio_dssim": float(dsim_val.item() / (l1_val.item() + dsim_val.item() + 1e-10)),
            "l1_grads": l1_grads,
            "dssim_grads": dsim_grads,
            "combined_grads": combined_grads,
        })

    # Average across cameras
    avg = {
        "n_gaussians": N,
        "avg_psnr": float(np.mean([r["psnr"] for r in results])),
        "avg_l1_loss": float(np.mean([r["l1_loss"] for r in results])),
        "avg_dssim_loss": float(np.mean([r["dssim_loss"] for r in results])),
        "avg_loss_ratio_dssim": float(np.mean([r["loss_ratio_dssim"] for r in results])),
    }

    # Per-parameter averages
    for name in PARAM_NAMES:
        l1_norms = [r["l1_grads"][name] for r in results]
        dsim_norms = [r["dssim_grads"][name] for r in results]
        combined_norms = [r["combined_grads"][name] for r in results]
        l1_pe = [r["l1_grads"][f"{name}_per_elem"] for r in results]
        dsim_pe = [r["dssim_grads"][f"{name}_per_elem"] for r in results]

        avg[f"l1_grad_{name}"] = float(np.mean(l1_norms))
        avg[f"dssim_grad_{name}"] = float(np.mean(dsim_norms))
        avg[f"combined_grad_{name}"] = float(np.mean(combined_norms))
        avg[f"l1_grad_{name}_pe"] = float(np.mean(l1_pe))
        avg[f"dssim_grad_{name}_pe"] = float(np.mean(dsim_pe))

        # Ratio: D-SSIM gradient / total gradient
        total = avg[f"l1_grad_{name}"] + avg[f"dssim_grad_{name}"]
        avg[f"grad_ratio_dssim_{name}"] = float(
            avg[f"dssim_grad_{name}"] / total if total > 0 else 0
        )

    # Overall gradient norms (sum across all params)
    total_l1 = sum(avg[f"l1_grad_{name}"] for name in PARAM_NAMES)
    total_dsim = sum(avg[f"dssim_grad_{name}"] for name in PARAM_NAMES)
    avg["total_l1_grad"] = total_l1
    avg["total_dssim_grad"] = total_dsim
    avg["total_grad_ratio_dssim"] = float(total_dsim / (total_l1 + total_dsim) if (total_l1 + total_dsim) > 0 else 0)

    return results, avg


def main():
    print("=" * 72)
    print("C42 Phase E-1: Gradient Redundancy Profiling")
    print("=" * 72)

    torch.manual_seed(42)
    repo_root = Path(__file__).resolve().parent.parent.parent

    print("\n  [Loading dataset...]")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)

    all_results = []
    all_avgs = []

    # ── Profile early training (mid_16 series: 1000, 2000, 3000) ──
    for it in MID_ITERATIONS:
        ckpt_path = repo_root / CKPT_MID.format(it)
        if not ckpt_path.exists():
            print(f"  iter {it}: not found, skipping")
            continue
        print(f"\n  Profiling iter {it}...", end=" ", flush=True)
        per_cam, avg = profile_checkpoint(ckpt_path, dataset, CAM_INDICES)
        all_results.append({"iteration": it, "series": "mid_16", "per_camera": per_cam})
        all_avgs.append({"iteration": it, "series": "mid_16", **avg})
        print(f"PSNR={avg['avg_psnr']:.2f}  "
              f"||g_L1||={avg['total_l1_grad']:.4f}  "
              f"||g_DSSIM||={avg['total_dssim_grad']:.4f}  "
              f"ratio={avg['total_grad_ratio_dssim']:.4f}")

    # ── Profile main training (v2_16 series: 5000-30000) ──
    for it in ITERATIONS:
        ckpt_path = repo_root / CKPT_BASE.format(it)
        if not ckpt_path.exists():
            print(f"  iter {it}: not found, skipping")
            continue
        print(f"\n  Profiling iter {it}...", end=" ", flush=True)
        per_cam, avg = profile_checkpoint(ckpt_path, dataset, CAM_INDICES)
        all_results.append({"iteration": it, "series": "v2_16", "per_camera": per_cam})
        all_avgs.append({"iteration": it, "series": "v2_16", **avg})
        print(f"PSNR={avg['avg_psnr']:.2f}  "
              f"||g_L1||={avg['total_l1_grad']:.4f}  "
              f"||g_DSSIM||={avg['total_dssim_grad']:.4f}  "
              f"ratio={avg['total_grad_ratio_dssim']:.4f}")

    # ── Summary table ──
    print(f"\n{'='*100}")
    print("GRADIENT REDUNDANCY SUMMARY")
    print(f"{'='*100}")
    print(f"\n  {'Iter':>6s}  {'GS':>8s}  {'PSNR':>6s}  "
          f"{'L1_loss':>8s}  {'DSSIM':>8s}  {'loss_r':>6s}  "
          f"{'||g_L1||':>10s}  {'||g_DS||':>10s}  {'g_ratio':>7s}  "
          f"{'g_xyz_r':>7s}  {'g_opa_r':>7s}  {'g_scl_r':>7s}  {'g_rot_r':>7s}  {'g_shs_r':>7s}")
    print("  " + "-" * 120)

    for a in all_avgs:
        print(f"  {a['iteration']:>6d}  {a['n_gaussians']:>8,}  {a['avg_psnr']:>6.2f}  "
              f"{a['avg_l1_loss']:>8.4f}  {a['avg_dssim_loss']:>8.4f}  {a['avg_loss_ratio_dssim']:>6.3f}  "
              f"{a['total_l1_grad']:>10.4f}  {a['total_dssim_grad']:>10.4f}  {a['total_grad_ratio_dssim']:>7.4f}  "
              f"{a['grad_ratio_dssim_xyz']:>7.4f}  {a['grad_ratio_dssim_opacity']:>7.4f}  "
              f"{a['grad_ratio_dssim_scales']:>7.4f}  {a['grad_ratio_dssim_rotations']:>7.4f}  "
              f"{a['grad_ratio_dssim_shs']:>7.4f}")

    # ── Temporal trend analysis ──
    print(f"\n{'='*72}")
    print("TEMPORAL TREND ANALYSIS")
    print(f"{'='*72}")

    if len(all_avgs) >= 2:
        first = all_avgs[0]
        last = all_avgs[-1]

        print(f"\n  From iter {first['iteration']} to iter {last['iteration']}:")
        print(f"    PSNR:                  {first['avg_psnr']:.2f} -> {last['avg_psnr']:.2f}")
        print(f"    L1 loss:               {first['avg_l1_loss']:.4f} -> {last['avg_l1_loss']:.4f}  "
              f"({(last['avg_l1_loss']/first['avg_l1_loss']-1)*100:+.1f}%)")
        print(f"    D-SSIM loss:           {first['avg_dssim_loss']:.4f} -> {last['avg_dssim_loss']:.4f}  "
              f"({(last['avg_dssim_loss']/first['avg_dssim_loss']-1)*100:+.1f}%)")
        print(f"    Loss ratio (D-SSIM):   {first['avg_loss_ratio_dssim']:.4f} -> {last['avg_loss_ratio_dssim']:.4f}")
        print(f"    ||grad_L1||:           {first['total_l1_grad']:.4f} -> {last['total_l1_grad']:.4f}  "
              f"({(last['total_l1_grad']/first['total_l1_grad']-1)*100:+.1f}%)")
        print(f"    ||grad_DSSIM||:        {first['total_dssim_grad']:.4f} -> {last['total_dssim_grad']:.4f}  "
              f"({(last['total_dssim_grad']/first['total_dssim_grad']-1)*100:+.1f}%)")
        print(f"    Grad ratio (D-SSIM):   {first['total_grad_ratio_dssim']:.4f} -> {last['total_grad_ratio_dssim']:.4f}")

        print(f"\n  Per-parameter gradient ratio (D-SSIM / total):")
        for name in PARAM_NAMES:
            key = f"grad_ratio_dssim_{name}"
            print(f"    {name:12s}: {first[key]:.4f} -> {last[key]:.4f}  "
                  f"({(last[key]/first[key]-1)*100 if first[key] > 0 else 0:+.1f}%)")

    # ── Research decision ──
    print(f"\n{'='*72}")
    print("RESEARCH DECISION")
    print(f"{'='*72}")

    # Check if D-SSIM gradient ratio decreases significantly
    if len(all_avgs) >= 2:
        ratios = [a["total_grad_ratio_dssim"] for a in all_avgs]
        first_ratio = ratios[0]
        last_ratio = ratios[-1]
        max_ratio = max(ratios)
        min_ratio = min(ratios)

        # "Significant decrease" = ratio drops by >30% relative
        relative_change = (last_ratio / first_ratio - 1) if first_ratio > 0 else 0

        print(f"\n  D-SSIM gradient ratio evolution:")
        print(f"    First (iter {all_avgs[0]['iteration']}): {first_ratio:.4f}")
        print(f"    Last  (iter {all_avgs[-1]['iteration']}): {last_ratio:.4f}")
        print(f"    Max:  {max_ratio:.4f}  Min: {min_ratio:.4f}")
        print(f"    Relative change: {relative_change*100:+.1f}%")

        if relative_change < -0.30:
            decision = "KEEP — D-SSIM gradient contribution decreases significantly (>30% relative drop)"
        elif relative_change < -0.15:
            decision = "MAYBE — D-SSIM gradient contribution decreases moderately (15-30% drop)"
        else:
            decision = "DROP — D-SSIM gradient contribution does not decrease significantly"

        print(f"\n  Decision: {decision}")
    else:
        decision = "INSUFFICIENT DATA"
        print(f"\n  Decision: {decision}")

    # ── Save ──
    output = {
        "experiment": "C42 Phase E-1: Gradient Redundancy Profiling",
        "config": {
            "scene": "room",
            "lambda_dssim": LAMBDA_DSSIM,
            "cameras": CAM_INDICES,
            "param_names": PARAM_NAMES,
        },
        "checkpoints": all_avgs,
        "per_camera_detail": all_results,
        "decision": decision,
    }

    save_path = Path("results/phase-c42/c42_e1_gradient_profile.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
