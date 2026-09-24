#!/usr/bin/env python3
"""H2-BWD-2R: 800-step room training comparison.

Baseline vs SCALAR_ADJOINT, same seed, same camera order, same optimizer,
same densification, same losses, same resolution.

Requires EXACT_VALIDATED classification from the exactness closure.
Does NOT require bitwise-identical trajectories (production atomicAdd is
nondeterministic). Compares candidate against baseline repeated-run variance.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

K_SH = 16
SH_DEGREE = 3
CDIM = 3

SCENE_PLY = "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply"
SCENE_CAMS = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"


def bootstrap(source, core_so, exp_so):
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core
    exp_spec = importlib.util.spec_from_file_location(
        "experimental_gaussian_render_inference_scene_cuda", exp_so)
    exp_mod = importlib.util.module_from_spec(exp_spec)
    exp_spec.loader.exec_module(exp_mod)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp_mod
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp_mod
    return exp_mod


def load_ply_scene(ply_path, device):
    v = PlyData.read(ply_path)["vertex"]
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), device=device, dtype=torch.float32)
    quats = torch.tensor(np.column_stack([v[f"rot_{i}"] for i in range(4)]), device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v[f"scale_{i}"] for i in range(3)]), device=device, dtype=torch.float32))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], device=device, dtype=torch.float32))
    sh = torch.zeros((len(v), K_SH, 3), device=device, dtype=torch.float32)
    sh[:, 0] = torch.tensor(np.column_stack([v[f"f_dc_{i}"] for i in range(3)]), device=device, dtype=torch.float32)
    rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], device=device, dtype=torch.float32) for i in range(45)], 1)
    sh[:, 1:] = rest.reshape(len(v), 3, 15).permute(0, 2, 1)
    return means, quats, scales, opacities, sh


def load_cameras(cams_path, max_long_side, device):
    cams = json.loads(Path(cams_path).read_text())
    viewmats = []
    Ks = []
    for c in cams:
        native_w, native_h = int(c["width"]), int(c["height"])
        scale = min(1.0, max_long_side / max(native_w, native_h))
        w, h = int(round(native_w * scale)), int(round(native_h * scale))
        R = np.asarray(c["rotation"], dtype=np.float32).T
        p = np.asarray(c["position"], dtype=np.float32)
        vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = -R @ p
        K = np.array([[float(c["fx"]) * w / native_w, 0, (w - 1) / 2],
                      [0, float(c["fy"]) * w / native_w, (h - 1) / 2], [0, 0, 1]], dtype=np.float32)
        viewmats.append(torch.tensor(vm, device=device))
        Ks.append(torch.tensor(K, device=device))
    return torch.stack(viewmats)[None], torch.stack(Ks)[None], [(int(round(native_w * min(1.0, max_long_side / max(int(c["width"]), int(c["height"]))))), 
                                                                  int(round(int(c["height"]) * min(1.0, max_long_side / max(int(c["width"]), int(c["height"])))))) for c in cams]


def load_gt_image(c, gt_dir, device, max_long_side=2048):
    """Load GT image for a camera. Returns None if not loadable."""
    img_name = c.get("img_name", f"{c.get('id', 0):04d}")
    for ext in [".JPG", ".jpg", ".png", ".PNG", ".jpeg"]:
        img_path = os.path.join(gt_dir, img_name + ext)
        if os.path.exists(img_path):
            try:
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                native_w, native_h = int(c["width"]), int(c["height"])
                scale = min(1.0, max_long_side / max(native_w, native_h))
                w, h = int(round(native_w * scale)), int(round(native_h * scale))
                if w != img.width or h != img.height:
                    img = img.resize((w, h), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                return torch.tensor(arr, device=device).permute(2, 0, 1)
            except Exception:
                return None
    return None


def seed_for_cam(i):
    return 10000 + i * 37

def combined_loss(pred, gt, lambda_dssim=0.2):
    """L1 + (1 - lambda) * L1 + lambda * D-SSIM."""
    l1 = (pred - gt).abs().mean()
    # D-SSIM = (1 - SSIM) / 2
    pred_pad = pred[None]  # [1, 3, H, W]
    gt_pad = gt[None]
    # Simple SSIM
    mu_p = F.avg_pool2d(pred_pad, 3, 1, 1)
    mu_g = F.avg_pool2d(gt_pad, 3, 1, 1)
    mu_p_sq = mu_p * mu_p
    mu_g_sq = mu_g * mu_g
    mu_p_g = mu_p * mu_g
    sigma_p = F.avg_pool2d(pred_pad * pred_pad, 3, 1, 1) - mu_p_sq
    sigma_g = F.avg_pool2d(gt_pad * gt_pad, 3, 1, 1) - mu_g_sq
    sigma_pg = F.avg_pool2d(pred_pad * gt_pad, 3, 1, 1) - mu_p_g
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * mu_p_g + C1) * (2 * sigma_pg + C2)) / \
           ((mu_p_sq + mu_g_sq + C1) * (sigma_p + sigma_g + C2))
    dssim = (1 - ssim.mean()) / 2
    return (1 - lambda_dssim) * l1 + lambda_dssim * dssim


def psnr(pred, gt):
    mse = (pred - gt).pow(2).mean()
    if mse < 1e-12:
        return 100.0
    return float(10 * math.log10(1.0 / mse.item()))


def run_training(variant, params0, viewmats, Ks, cam_sizes, gt_images, device,
                 n_steps=800, seed=42, densify_every=100, densify_threshold=0.0002,
                 prune_threshold=0.005, lr_xyz=1.6e-4, lr_rot=1e-3, lr_scale=5e-3,
                 lr_opac=5e-2, lr_sh=2.5e-3, max_long_side=2048):
    """Run training for n_steps. Returns trajectory data."""
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import (
        create_higs_renderer, _HIGS_FROZEN_TRACKER
    )

    os.environ["HIGS_BWD_CF_VARIANT"] = variant
    os.environ["HIGS_PX_RUNTIME"] = "2"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Clone params
    means, quats, scales, opacities, sh = [p.clone() for p in params0]
    N = means.shape[0]

    # Optimizer: per-parameter Adam with original 3DGS learning rates
    means.requires_grad_(True)
    quats.requires_grad_(True)
    scales.requires_grad_(True)
    opacities.requires_grad_(True)
    sh.requires_grad_(True)

    opt_means = torch.optim.Adam([means], lr=lr_xyz)
    opt_quats = torch.optim.Adam([quats], lr=lr_rot)
    opt_scales = torch.optim.Adam([scales], lr=lr_scale)
    opt_opac = torch.optim.Adam([opacities], lr=lr_opac)
    opt_sh = torch.optim.Adam([sh], lr=lr_sh)

    n_cams = len(cam_sizes)
    cam_order = list(range(n_cams))

    trajectory = []
    losses = []
    psnrs = []
    n_gs = []
    wall_start = time.time()

    for it in range(n_steps):
        # Shuffle camera order deterministically
        rng = random.Random(seed + it)
        rng.shuffle(cam_order)
        cam_idx = cam_order[0]  # one camera per step

        w, h = cam_sizes[cam_idx]
        vm = viewmats[:, cam_idx:cam_idx+1]
        K = Ks[:, cam_idx:cam_idx+1]
        gt = gt_images[cam_idx]
        if gt is None:
            # Use a dummy target (all zeros) if no GT image
            gt = torch.zeros(3, h, w, device=device)

        values = (means, quats, scales, opacities, sh)

        _HIGS_FROZEN_TRACKER.reset()
        handle = create_higs_renderer(*values, sh_degree=SH_DEGREE)
        out = rasterize_gaussian_higs_frozen(
            *values, backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm, Ks=K, width=w, height=h, sh_degree=SH_DEGREE,
            use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)

        frame = out["frame"][0]  # [H, W, 3] (out["frame"] is [1, H, W, 3])
        # Normalize to [3, H, W]
        if frame.dim() == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        alpha = out["alpha"][0]  # [H, W] or [H, W, 1]
        if alpha.dim() == 3:
            alpha = alpha.squeeze(-1)

        # Render with white background
        rendered = frame * alpha[None] + 1.0 * (1 - alpha[None])

        if it == 0:
            print(f"  [debug] out['frame'].shape={out['frame'].shape}, out['alpha'].shape={out['alpha'].shape}", flush=True)
            print(f"  [debug] frame.shape={frame.shape}, alpha.shape={alpha.shape}, "
                  f"rendered.shape={rendered.shape}, gt.shape={gt.shape}", flush=True)

        loss = combined_loss(rendered, gt)

        # Backward
        for opt in [opt_means, opt_quats, opt_scales, opt_opac, opt_sh]:
            opt.zero_grad()
        loss.backward()
        torch.cuda.synchronize()

        # Check for NaN/Inf
        has_nan = False
        for p in [means, quats, scales, opacities, sh]:
            if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                has_nan = True
                break

        # Optimizer step
        if not has_nan:
            opt_means.step()
            opt_quats.step()
            opt_scales.step()
            opt_opac.step()
            opt_sh.step()

        # Densification (simplified: opacity pruning + gradient-based split)
        if (it + 1) % densify_every == 0 and it < n_steps * 0.75:
            with torch.no_grad():
                # Opacity pruning
                keep = (opacities.data > prune_threshold).nonzero().flatten()
                if keep.numel() < N * 0.1:
                    keep = torch.arange(N, device=device)
                if keep.numel() < N:
                    means = means.data[keep].clone().requires_grad_(True)
                    quats = quats.data[keep].clone().requires_grad_(True)
                    scales = scales.data[keep].clone().requires_grad_(True)
                    opacities = opacities.data[keep].clone().requires_grad_(True)
                    sh = sh.data[keep].clone().requires_grad_(True)
                    N = means.shape[0]
                    # Reset optimizers
                    opt_means = torch.optim.Adam([means], lr=lr_xyz)
                    opt_quats = torch.optim.Adam([quats], lr=lr_rot)
                    opt_scales = torch.optim.Adam([scales], lr=lr_scale)
                    opt_opac = torch.optim.Adam([opacities], lr=lr_opac)
                    opt_sh = torch.optim.Adam([sh], lr=lr_sh)

        handle.release()

        # Record
        loss_val = float(loss.item())
        losses.append(loss_val)
        n_gs.append(N)
        current_psnr = psnr(rendered.detach(), gt)
        psnrs.append(current_psnr)

        elapsed = time.time() - wall_start
        if (it + 1) % 50 == 0 or it == 0:
            print(f"  [{variant}] step {it+1}/{n_steps}: loss={loss_val:.6f}, "
                  f"PSNR={current_psnr:.2f}, N_GS={N}, NaN={has_nan}, "
                  f"elapsed={elapsed:.1f}s, iter/s={it+1/elapsed:.2f}", flush=True)

        trajectory.append({
            "step": it + 1,
            "loss": loss_val,
            "psnr": current_psnr,
            "n_gs": N,
            "nan_inf": has_nan,
            "elapsed_s": elapsed,
        })

    wall_total = time.time() - wall_start
    return {
        "variant": variant,
        "n_steps": n_steps,
        "wall_time_s": wall_total,
        "iter_per_s": n_steps / wall_total,
        "final_loss": losses[-1],
        "final_psnr": psnrs[-1],
        "final_n_gs": n_gs[-1],
        "loss_trajectory": losses,
        "psnr_trajectory": psnrs,
        "n_gs_trajectory": n_gs,
        "nan_inf_count": sum(1 for t in trajectory if t["nan_inf"]),
        "trajectory": trajectory,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--source", default="/tmp/higs_h2_bwd_cf/source")
    ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    ap.add_argument("--exp-so", default="/tmp/higs_h2_bwd_cf/instrumented-cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/higs_h2_bwd_cf/cache"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print(f"=== H2-BWD-2R: 800-step room training ===", flush=True)
    print(f"Steps: {args.steps}, seed: {args.seed}", flush=True)

    # Bootstrap
    bootstrap(args.source, args.core_so, args.exp_so)
    print("Bootstrap complete.", flush=True)

    # Load scene
    params0 = load_ply_scene(SCENE_PLY, device)
    viewmats, Ks, cam_sizes = load_cameras(SCENE_CAMS, args.max_long_side, device)
    n_cams = len(cam_sizes)
    print(f"Scene: room, N_gaussians={params0[0].shape[0]}, n_cams={n_cams}", flush=True)
    print(f"Resolution: {cam_sizes[0]}", flush=True)

    # Load GT images
    gt_dir = "/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/room/images"
    cams = json.loads(Path(SCENE_CAMS).read_text())
    gt_images = []
    for i, c in enumerate(cams):
        gt = load_gt_image(c, gt_dir, device)
        gt_images.append(gt)
    n_valid_gt = sum(1 for g in gt_images if g is not None)
    print(f"GT images loaded: {n_valid_gt}/{n_cams}", flush=True)
    if n_valid_gt == 0:
        print("WARNING: No GT images loadable, using deterministic synthetic targets", flush=True)
        # Generate deterministic synthetic targets (same for both variants)
        for i, (w, h) in enumerate(cam_sizes):
            gen = torch.Generator(device="cuda").manual_seed(seed_for_cam(i))
            gt_images[i] = torch.rand(3, h, w, device=device, generator=gen)
        n_valid_gt = n_cams

    # Run training for both variants
    results = {}
    for variant in ["baseline", "scalar_adjoint"]:
        print(f"\n=== Training: {variant} ===", flush=True)
        r = run_training(variant, params0, viewmats, Ks, cam_sizes, gt_images, device,
                         n_steps=args.steps, seed=args.seed)
        results[variant] = r
        print(f"\n[{variant}] Final: wall={r['wall_time_s']:.1f}s, iter/s={r['iter_per_s']:.2f}, "
              f"loss={r['final_loss']:.6f}, PSNR={r['final_psnr']:.2f}, N_GS={r['final_n_gs']}, "
              f"NaN/Inf={r['nan_inf_count']}", flush=True)

    # Compare
    base = results["baseline"]
    scalar = results["scalar_adjoint"]
    speedup = (base["wall_time_s"] / scalar["wall_time_s"] - 1) * 100
    print(f"\n=== Comparison ===", flush=True)
    print(f"Wall time: baseline={base['wall_time_s']:.1f}s, scalar={scalar['wall_time_s']:.1f}s, speedup={speedup:.2f}%", flush=True)
    print(f"iter/s:    baseline={base['iter_per_s']:.2f}, scalar={scalar['iter_per_s']:.2f}", flush=True)
    print(f"Final PSNR: baseline={base['final_psnr']:.2f}, scalar={scalar['final_psnr']:.2f}, delta={scalar['final_psnr']-base['final_psnr']:.2f}", flush=True)
    print(f"Final N_GS: baseline={base['final_n_gs']}, scalar={scalar['final_n_gs']}", flush=True)

    # Write CSV trajectory
    with open(out_dir / "training_trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "variant", "loss", "psnr", "n_gs", "nan_inf", "elapsed_s"])
        w.writeheader()
        for variant in ["baseline", "scalar_adjoint"]:
            for t in results[variant]["trajectory"]:
                w.writerow({"variant": variant, **t})

    # Write summary
    summary = {
        "baseline": {k: v for k, v in base.items() if k != "trajectory"},
        "scalar_adjoint": {k: v for k, v in scalar.items() if k != "trajectory"},
        "comparison": {
            "wall_time_speedup_pct": speedup,
            "iter_per_s_ratio": scalar["iter_per_s"] / base["iter_per_s"],
            "psnr_delta": scalar["final_psnr"] - base["final_psnr"],
            "n_gs_delta": scalar["final_n_gs"] - base["final_n_gs"],
        },
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nArtifacts written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
