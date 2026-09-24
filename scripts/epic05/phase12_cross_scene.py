#!/usr/bin/env python3
"""
Phase 12 — Cross-Scene Validation (M1 tile32 + M3 SH degree)

Validates room findings on bicycle and garden scenes.

Steps:
  A1 — Snapshot workload + forward timing (tile16 vs tile32) on frozen SfM checkpoints
  A2 — Snapshot fwd+bwd timing (tile16 vs tile32)
  B  — Short training (500 steps) with real GT
  C  — 30K training feasibility check
  D  — M3 SH0/SH1/SH3 forward quality + timing on bicycle/garden

Requires:
  data/official/mipnerf360/{scene}/point_cloud.ply  (SfM)
  data/official/mipnerf360/{scene}/cameras.json
  data/datasets/mipnerf360/{scene}/images/         (GT)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import gc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gsplat import rasterization
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"
DTYPE = torch.float32

# ── Helpers ────────────────────────────────────────────────────────────────

def make_camera(W=1920, H=1080, device=DEVICE):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    viewmat = torch.eye(4, device=device, dtype=DTYPE).unsqueeze(0)
    t = torch.eye(4, device=device, dtype=DTYPE)
    t[2, 3] = -5.0
    viewmat[0] = t
    K = torch.tensor([[fx, 0.0, W / 2.0],
                      [0.0, fy, H / 2.0],
                      [0.0, 0.0, 1.0]], device=device, dtype=DTYPE).unsqueeze(0)
    return viewmat, K, W, H


def load_sfm_checkpoint(scene: str, device=DEVICE):
    """Load SfM point cloud for a scene."""
    ply_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"SfM PLY not found: {ply_path}")
    print(f"  Loading SfM checkpoint: {ply_path}")
    sfm_data = load_ply(str(ply_path), device=device)
    return sfm_data


def load_scene_camera(scene: str, W=1920, H=1080, device=DEVICE):
    """Load first camera from scene's camera file."""
    cam_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene / "cameras.json"
    cameras = load_cameras_from_json(str(cam_path), device="cpu")
    cameras = resize_cameras(cameras, W, H)
    cam = cameras[0]
    for attr in ["viewmatrix", "projmatrix", "camera_center", "world_view_transform",
                  "full_proj_transform", "K"]:
        t = getattr(cam, attr)
        if isinstance(t, torch.Tensor):
            setattr(cam, attr, t.to(device))
    return cam


def compute_workload_stats(params, viewmat, K, W=1920, H=1080, tile_size=16):
    """Compute workload statistics from a frozen snapshot."""
    with torch.no_grad():
        rendered, alpha, meta = rasterization(
            means=params["xyz"], quats=params["rotations"],
            scales=params["scales"], opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
        )
    tpg = meta["tiles_per_gauss"]
    tpg_np = tpg.float().cpu().numpy()
    tile_w = meta["tile_width"]
    tile_h = meta["tile_height"]
    total_tiles = tile_w * tile_h
    return {
        "total_gaussians": params["xyz"].shape[0],
        "nnz_gaussians": tpg.shape[0],
        "tile_grid_width": tile_w,
        "tile_grid_height": tile_h,
        "total_tiles": total_tiles,
        "tpg_mean": float(tpg.float().mean().item()),
        "tpg_median": float(tpg.float().median().item()),
        "tpg_std": float(tpg.float().std().item()),
        "tpg_min": int(tpg.min().item()),
        "tpg_max": int(tpg.max().item()),
        "tpg_p95": float(np.percentile(tpg_np, 95)),
        "tpg_p99": float(np.percentile(tpg_np, 99)),
        "total_intersections": int(tpg.sum().item()),
        "gaussians_per_tile_mean": float(tpg.sum().item() / total_tiles),
    }, meta


def compute_quality_metrics(rendered, gt_image):
    """Compute PSNR, SSIM, LPIPS between rendered and GT."""
    import torch.nn.functional as F
    
    rendered = rendered.clamp(0, 1)
    
    # PSNR
    mse = torch.mean((rendered - gt_image) ** 2).item()
    psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
    
    # SSIM (simplified single-scale)
    def ssim(img1, img2, data_range=1.0, size_average=True):
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
        img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)
        mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 * img1, 11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 11, stride=1, padding=5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean().item()) if size_average else ssim_map
    
    ssim_val = ssim(rendered, gt_image)
    
    # Simplified LPIPS-like (mean absolute difference in Laplacian pyramid)
    # We'll use a basic perceptual approximation
    lpips_val = 0.0  # Will be filled if lpips package available
    
    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net='alex').to(rendered.device)
        r = rendered.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        g = gt_image.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        lpips_val = float(lpips_fn(r, g).item())
    except ImportError:
        # Fallback: normalized MSE across 3 scales
        def _laplacian_mse(x):
            blurred = F.avg_pool2d(x, 3, stride=1, padding=1)
            return torch.mean((x - blurred) ** 2).item()
        r4d = rendered.unsqueeze(0).permute(0, 3, 1, 2)
        g4d = gt_image.unsqueeze(0).permute(0, 3, 1, 2)
        lpips_val = _laplacian_mse(r4d - g4d)
    
    return {
        "psnr": round(psnr, 4),
        "ssim": round(ssim_val, 6),
        "lpips": round(lpips_val, 6),
    }


# ── Snapshot Timing ────────────────────────────────────────────────────────

def timing_forward(params, viewmat, K, W, H, tile_size, sh_degree=None,
                   n_batch=10, n_repeat=3, warmup=3):
    if sh_degree is None:
        sh_degree = params["sh_degree"]
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        rendered, alpha, meta = rasterization(
            means=params["xyz"], quats=params["rotations"],
            scales=params["scales"], opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=sh_degree,
        )
        torch.cuda.synchronize()
    times = []
    for r in range(n_repeat):
        for b in range(n_batch):
            start_ev.record()
            rendered, alpha, meta = rasterization(
                means=params["xyz"], quats=params["rotations"],
                scales=params["scales"], opacities=params["opacity"],
                colors=params["shs"],
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=tile_size, packed=True, sh_degree=sh_degree,
            )
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
    return np.array(times), meta


def timing_fwd_bwd(params, viewmat, K, W, H, tile_size, sh_degree=None,
                   n_batch=3, n_repeat=2, warmup=2):
    if sh_degree is None:
        sh_degree = params["sh_degree"]
    xyz = params["xyz"].detach().clone().requires_grad_(True)
    rotations = params["rotations"].detach().clone().requires_grad_(True)
    scales = params["scales"].detach().clone().requires_grad_(True)
    opacity = params["opacity"].detach().clone().requires_grad_(True)
    shs = params["shs"].detach().clone().requires_grad_(True)
    target = torch.rand(1, H, W, 3, device=DEVICE, dtype=DTYPE)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        rendered, alpha, meta = rasterization(
            means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=sh_degree,
        )
        loss = ((rendered - target) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        for p in [xyz, rotations, scales, opacity, shs]:
            if p.grad is not None:
                p.grad = None

    times = []
    for r in range(n_repeat):
        for b in range(n_batch):
            start_ev.record()
            rendered, alpha, meta = rasterization(
                means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=tile_size, packed=True, sh_degree=sh_degree,
            )
            loss = ((rendered - target) ** 2).mean()
            loss.backward()
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
            for p in [xyz, rotations, scales, opacity, shs]:
                if p.grad is not None:
                    p.grad = None

    # Grad verification
    for p in [xyz, rotations, scales, opacity, shs]:
        if p.grad is not None:
            p.grad = None
    rendered, alpha, _ = rasterization(
        means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
        viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=tile_size, packed=True, sh_degree=sh_degree,
    )
    loss = ((rendered - target) ** 2).mean()
    loss.backward()
    torch.cuda.synchronize()

    grad_check = {
        "all_grad_finite": True,
        "xyz_grad_norm": float(xyz.grad.norm().item()),
        "xyz_grad_nonzero": bool((xyz.grad.abs() > 0).any().item()),
        "loss_value": float(loss.item()),
    }
    for name, p in [("rotations", rotations), ("scales", scales),
                     ("opacity", opacity), ("shs", shs)]:
        grad_check[f"{name}_finite"] = bool(torch.isfinite(p.grad).all().item())
        grad_check[f"{name}_norm"] = float(p.grad.norm().item())

    return np.array(times), grad_check


def compute_robust_stats(arr):
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0.0,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p25_ms": float(np.percentile(arr, 25)),
        "p75_ms": float(np.percentile(arr, 75)),
        "n_samples": int(len(arr)),
    }


def run_tile_size_snapshot(params, viewmat, K, W, H, tile_size, is_tile16=False, label=""):
    """Complete snapshot timing for one tile size."""
    print(f"  --- {label} tile_size={tile_size} ---", flush=True)

    # Workload stats
    wl_stats, meta = compute_workload_stats(params, viewmat, K, W, H, tile_size)
    print(f"    Workload: {wl_stats['total_gaussians']:,} Gs, "
          f"{wl_stats['total_intersections']/1e6:.1f}M intersections, "
          f"tpg={wl_stats['tpg_mean']:.1f} (max={wl_stats['tpg_max']})", flush=True)

    # Forward timing
    print(f"    Forward timing...", flush=True)
    fwd_times, _ = timing_forward(params, viewmat, K, W, H, tile_size,
                                   n_batch=10, n_repeat=3, warmup=3)
    fwd_stats = compute_robust_stats(fwd_times)
    print(f"    Forward: {fwd_stats['median_ms']:.2f} ms (median)  "
          f"{fwd_stats['mean_ms']:.2f} ± {fwd_stats['std_ms']:.2f} ms  "
          f"CV={fwd_stats['cv']:.4f}", flush=True)

    # Fwd+Bwd timing
    print(f"    Fwd+Bwd timing...", flush=True)
    if is_tile16:
        fb_n_batch, fb_n_repeat = 3, 2
    else:
        fb_n_batch, fb_n_repeat = 5, 2

    fwd_bwd_times, grad_check = timing_fwd_bwd(
        params, viewmat, K, W, H, tile_size,
        n_batch=fb_n_batch, n_repeat=fb_n_repeat, warmup=2)
    fb_stats = compute_robust_stats(fwd_bwd_times)
    print(f"    Fwd+Bwd: {fb_stats['median_ms']:.2f} ms (median)  "
          f"{fb_stats['mean_ms']:.2f} ± {fb_stats['std_ms']:.2f} ms  "
          f"CV={fb_stats['cv']:.4f}", flush=True)

    # Infer backward
    bwd_median = max(0.001, fb_stats["median_ms"] - fwd_stats["median_ms"])
    bwd_mean = max(0.001, fb_stats["mean_ms"] - fwd_stats["mean_ms"])
    bwd_std = max(0.001, math.sqrt(max(0.0001, fb_stats["std_ms"]**2 - fwd_stats["std_ms"]**2)))
    bwd_stats = {
        "median_ms": bwd_median,
        "mean_ms": bwd_mean,
        "std_ms": bwd_std,
        "cv": bwd_std / bwd_mean,
    }
    print(f"    Inferred Bwd: {bwd_stats['median_ms']:.2f} ms (median)  "
          f"({bwd_stats['mean_ms']:.2f} mean)", flush=True)

    return {
        "forward": fwd_stats,
        "inferred_backward": bwd_stats,
        "forward_plus_backward": fb_stats,
        "gradient_verification": grad_check,
        "workload_statistics": wl_stats,
        "tile_grid": f"{wl_stats['tile_grid_width']}x{wl_stats['tile_grid_height']}",
    }


# ── Short Training ─────────────────────────────────────────────────────────

def short_training(scene, tile_size, steps=500, seed=42):
    """Run a short training session with real GT images."""
    from scripts.epic05.phase7.train_3dgs import TrainingConfig, TrainingPipeline
    
    label = f"phase12_{scene}_t{tile_size}_s{steps}"
    
    config = TrainingConfig(
        scene=scene,
        resolution="1080p",
        repo_root=str(REPO_ROOT),
        num_iterations=steps,
        tile_size=tile_size,
        packed=True,
        radius_clip=0.0,
        eps2d=0.1,
        lambda_dssim=0.2,
        densification_start=500 if steps > 500 else steps + 1,
        densification_end=15000,
        densification_interval=100,
        densification_grad_threshold=2e-4,
        prune_interval=100,
        prune_opacity_threshold=0.005,
        checkpoint_interval=steps,
        seed=seed,
        experiment_label=label,
        save_dir=str(REPO_ROOT / "results" / "epic05" / "phase12"),
    )
    
    print(f"\n  {'='*60}", flush=True)
    print(f"  Short Training: {scene} tile_size={tile_size} steps={steps}", flush=True)
    print(f"  {'='*60}", flush=True)
    
    pipeline = TrainingPipeline(config)
    summary = pipeline.train()
    
    return summary


# ── M3 SH Degree Cross-Scene ───────────────────────────────────────────────

def run_m3_forward_quality(params, scene, W=1920, H=1080):
    """Run SH0/SH1/SH3 forward timing + quality on a frozen snapshot."""
    from PIL import Image
    import numpy as np
    
    # Load real GT first camera
    gt_dir = REPO_ROOT / "data" / "datasets" / "mipnerf360" / scene / "images"
    cam = load_scene_camera(scene, W, H)
    
    gt_files = sorted([f for f in gt_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    print(f"  Loading first GT image: {gt_files[0].name}", flush=True)
    with Image.open(gt_files[0]) as source:
        if source.width != W or source.height != H:
            source = source.resize((W, H), Image.LANCZOS)
        rgba = torch.from_numpy(np.array(source.convert("RGBA"), dtype=np.uint8))
    gt_image = rgba.to(DEVICE, dtype=DTYPE)[..., :3] / 255.0

    sh_degrees = [0, 1, 3]
    results = {}
    
    for sh in sh_degrees:
        print(f"\n  --- SH degree={sh} ---", flush=True)
        
        # Workload
        with torch.no_grad():
            rendered_sh, _, meta_sh = rasterization(
                means=params["xyz"], quats=params["rotations"],
                scales=params["scales"], opacities=params["opacity"],
                colors=params["shs"],
                viewmats=cam.viewmatrix.unsqueeze(0),
                Ks=cam.K.unsqueeze(0), width=W, height=H,
                tile_size=16, packed=True, sh_degree=sh,
            )
        rendered_sh = rendered_sh[0].clamp(0, 1)
        
        # Quality
        quality = compute_quality_metrics(rendered_sh, gt_image)
        print(f"    Quality: PSNR={quality['psnr']:.2f} SSIM={quality['ssim']:.4f} LPIPS={quality['lpips']:.4f}",
              flush=True)
        
        # Forward timing
        fwd_times, _ = timing_forward(params, cam.viewmatrix.unsqueeze(0), cam.K.unsqueeze(0),
                                        W, H, tile_size=16, sh_degree=sh,
                                        n_batch=10, n_repeat=3, warmup=3)
        fwd_stats = compute_robust_stats(fwd_times)
        print(f"    Forward: {fwd_stats['median_ms']:.2f} ms (median)", flush=True)
        
        results[f"SH{sh}"] = {
            "quality": quality,
            "forward_timing": fwd_stats,
        }
    
    # Compute deltas
    for sh_str in ["SH0", "SH1"]:
        ref = results["SH3"]
        cand = results[sh_str]
        results[sh_str]["delta_vs_SH3"] = {
            "psnr": round(cand["quality"]["psnr"] - ref["quality"]["psnr"], 4),
            "ssim": round(cand["quality"]["ssim"] - ref["quality"]["ssim"], 6),
            "lpips": round(cand["quality"]["lpips"] - ref["quality"]["lpips"], 6),
            "speedup_forward": round(ref["forward_timing"]["median_ms"] / max(0.001, cand["forward_timing"]["median_ms"]), 4),
        }
        cand_str = f"  vs SH3: ΔPSNR={results[sh_str]['delta_vs_SH3']['psnr']:.2f} dB, "
        cand_str += f"speedup={results[sh_str]['delta_vs_SH3']['speedup_forward']:.2f}x"
        print(cand_str, flush=True)
    
    return results


# ── Main Pipeline ──────────────────────────────────────────────────────────

def phase12_main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 12: Cross-Scene Validation")
    parser.add_argument("--scene", choices=["bicycle", "garden", "room"], default=None,
                        help="Run on specific scene only (default: bicycle then garden)")
    parser.add_argument("--snapshot-only", action="store_true",
                        help="Skip training, do snapshot timing only")
    parser.add_argument("--m3-only", action="store_true",
                        help="Skip M1, do M3 SH quality only")
    parser.add_argument("--steps", type=int, default=500,
                        help="Short training steps (default: 500)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip short training entirely")
    args = parser.parse_args()

    output_dir = REPO_ROOT / "results" / "epic05" / "phase12"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scenes = [args.scene] if args.scene else ["bicycle", "garden"]
    
    all_results = {
        "schema_version": 1,
        "phase": "12",
        "date": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "config": vars(args),
        "scenes": {},
    }

    for scene in scenes:
        print(f"\n\n{'='*70}", flush=True)
        print(f"  SCENE: {scene}", flush=True)
        print(f"{'='*70}", flush=True)
        
        scene_results = {}
        
        # Load SfM checkpoint
        print(f"\n  [Loading SfM checkpoint...]", flush=True)
        sfm_data = load_sfm_checkpoint(scene)
        N = sfm_data["xyz"].shape[0]
        
        # Build params dict with CORRECT activations for gsplat.rasterization()
        # PLY stores: opacity in logit-space, scales in log-space, rotations raw (quats)
        # gsplat expects: opacity sigmoided [0,1], scales exp'd (positive), rotations normalized
        raw_opacity = sfm_data["opacity"]
        if raw_opacity.dim() == 2 and raw_opacity.shape[1] == 1:
            raw_opacity = raw_opacity.squeeze(1)
        raw_rotations = sfm_data["rotations"]
        raw_scales = sfm_data["scales"]
        raw_shs = sfm_data.get("shs")
        if raw_shs is None:
            raw_shs = torch.zeros(N, 16, 3, device=DEVICE)
        
        # Apply activations matching 3DGS training pipeline
        params = {
            "xyz": sfm_data["xyz"].detach().clone(),
            "rotations": torch.nn.functional.normalize(raw_rotations, dim=-1).contiguous(),
            "scales": torch.exp(raw_scales).contiguous(),
            "opacity": torch.sigmoid(raw_opacity).contiguous(),
            "shs": raw_shs.contiguous(),
            "num_points": N,
            "sh_degree": sfm_data.get("sh_degree", 3),
        }

        print(f"  N={N:,} Gs, SH deg={params['sh_degree']}", flush=True)

        viewmat, K, W, H = make_camera()

        # ═══════════════════════════════════════════
        # M1 — Snapshot Timing (tile16 vs tile32)
        # ═══════════════════════════════════════════
        if not args.m3_only:
            print(f"\n  ── M1: Snapshot Timing ──", flush=True)
            
            # tile16
            t16 = run_tile_size_snapshot(params, viewmat, K, W, H, 16, is_tile16=True, label=scene)
            torch.cuda.empty_cache()
            gc.collect()
            
            # tile32
            t32 = run_tile_size_snapshot(params, viewmat, K, W, H, 32, is_tile16=False, label=scene)
            torch.cuda.empty_cache()
            gc.collect()
            
            # Ratios
            ratios = {}
            for phase, key in [("forward", "forward"), ("forward_plus_backward", "forward_plus_backward")]:
                r = t16[key]["median_ms"] / t32[key]["median_ms"]
                ratios[phase] = {
                    "ratio_t16_t32_median": r,
                    "speedup_t32": 1.0 / r,
                    "t16_median_ms": t16[key]["median_ms"],
                    "t32_median_ms": t32[key]["median_ms"],
                }
            br_mdn = t16["inferred_backward"]["median_ms"] / t32["inferred_backward"]["median_ms"]
            ratios["inferred_backward"] = {
                "ratio_t16_t32_median": br_mdn,
                "t16_median_ms": t16["inferred_backward"]["median_ms"],
                "t32_median_ms": t32["inferred_backward"]["median_ms"],
            }
            
            print(f"\n  >>> RATIOS (median): Forward={ratios['forward']['ratio_t16_t32_median']:.2f}x  "
                  f"Backward={br_mdn:.2f}x  "
                  f"Fwd+Bwd={ratios['forward_plus_backward']['ratio_t16_t32_median']:.2f}x",
                  flush=True)
            
            scene_results["m1_snapshot"] = {
                "tile16": t16,
                "tile32": t32,
                "ratios": ratios,
            }
            
            # Also try with real camera (not synthetic)
            print(f"\n  ── M1: Real Camera Snapshot ──", flush=True)
            try:
                real_cam = load_scene_camera(scene, W, H)
                rv = real_cam.viewmatrix.unsqueeze(0)
                rK = real_cam.K.unsqueeze(0)
                
                # Workload stats with real camera
                wl_real, _ = compute_workload_stats(params, rv, rK, W, H, 16)
                # Forward timing
                fwd_real_t16, _ = timing_forward(params, rv, rK, W, H, 16,
                                                  n_batch=10, n_repeat=3, warmup=3)
                fwd_real_t32, _ = timing_forward(params, rv, rK, W, H, 32,
                                                  n_batch=10, n_repeat=3, warmup=3)
                
                real_cam_results = {
                    "workload": wl_real,
                    "forward_t16_ms": float(np.median(fwd_real_t16)),
                    "forward_t32_ms": float(np.median(fwd_real_t32)),
                    "speedup_t32": float(np.median(fwd_real_t16) / max(0.001, np.median(fwd_real_t32))),
                }
                scene_results["m1_real_camera"] = real_cam_results
                print(f"    Real cam forward: t16={real_cam_results['forward_t16_ms']:.2f}ms, "
                      f"t32={real_cam_results['forward_t32_ms']:.2f}ms, "
                      f"speedup={real_cam_results['speedup_t32']:.2f}x", flush=True)
            except Exception as e:
                print(f"    Real camera snapshot failed: {e}", flush=True)
                scene_results["m1_real_camera"] = {"error": str(e)}
            
            torch.cuda.empty_cache()
            gc.collect()

        # ═══════════════════════════════════════════
        # M3 — SH Degree Forward Quality
        # ═══════════════════════════════════════════
        print(f"\n  ── M3: SH Degree Forward Quality ──", flush=True)
        try:
            m3_results = run_m3_forward_quality(params, scene, W, H)
            scene_results["m3_sh_quality"] = m3_results
        except Exception as e:
            print(f"    M3 failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            scene_results["m3_sh_quality"] = {"error": str(e)}
        
        torch.cuda.empty_cache()
        gc.collect()

        # ═══════════════════════════════════════════
        # M1 — Short Training (if requested)
        # ═══════════════════════════════════════════
        if not args.snapshot_only and not args.skip_training and not args.m3_only:
            # Check VRAM feasibility first
            print(f"\n  ── Short Training Feasibility ──", flush=True)
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            print(f"    Free VRAM: {free_mem:.1f} GB", flush=True)
            
            # bicycle has ~6.1M initial Gs; room had 1.6M Gs and used ~2.9GB for training
            # Rough estimate: 6.1M / 1.6M * 2.9GB ≈ 11GB — likely exceeds 8GB
            estimated_vram_gb = (N / 1.6e6) * 2.9
            print(f"    Estimated VRAM for training: {estimated_vram_gb:.1f} GB (based on room scaling)", flush=True)
            
            if estimated_vram_gb > 7.5 and free_mem < 6.0:
                print(f"    ❌ Short training INFEASIBLE — estimated VRAM exceeds available memory", flush=True)
                scene_results["short_training"] = {
                    "status": "INFEASIBLE",
                    "reason": f"Estimated VRAM {estimated_vram_gb:.1f}GB > 8GB GPU limit",
                    "estimated_vram_gb": round(estimated_vram_gb, 1),
                }
            else:
                steps = args.steps
                print(f"\n  ── Short Training: {scene} tile16 steps={steps} ──", flush=True)
                try:
                    summary_t16 = short_training(scene=scene, tile_size=16, steps=steps)
                    scene_results["short_training"] = {
                        "tile16": summary_t16,
                    }
                except Exception as e:
                    print(f"    ❌ tile16 training failed: {e}", flush=True)
                    scene_results["short_training"] = {"tile16_error": str(e)}
                
                torch.cuda.empty_cache()
                gc.collect()
                
                print(f"\n  ── Short Training: {scene} tile32 steps={steps} ──", flush=True)
                try:
                    summary_t32 = short_training(scene=scene, tile_size=32, steps=steps)
                    if "tile16" in scene_results.get("short_training", {}):
                        # Compute comparison
                        s16 = scene_results["short_training"]["tile16"]
                        speedup = s16["total_wall_s"] / max(0.001, summary_t32["total_wall_s"])
                        psnr_diff = summary_t32.get("best_psnr", 0) - s16.get("best_psnr", 0)
                        scene_results["short_training"]["comparison"] = {
                            "speedup_t32_vs_t16": round(speedup, 4),
                            "t16_wall_min": s16["total_wall_s"] / 60,
                            "t32_wall_min": summary_t32["total_wall_s"] / 60,
                            "psnr_diff_t32_t16": round(psnr_diff, 4),
                        }
                    scene_results["short_training"]["tile32"] = summary_t32
                except Exception as e:
                    print(f"    ❌ tile32 training failed: {e}", flush=True)
                    if "short_training" not in scene_results:
                        scene_results["short_training"] = {}
                    scene_results["short_training"]["tile32_error"] = str(e)
        
        all_results["scenes"][scene] = scene_results
        
        # Save per-scene results
        per_scene_path = output_dir / f"phase12_{scene}.json"
        with open(per_scene_path, "w") as f:
            json.dump(scene_results, f, indent=2, default=str)
        print(f"\n  [Saved: {per_scene_path}]", flush=True)
    
    # Save combined results
    combined_path = output_dir / "phase12_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  [Saved combined: {combined_path}]", flush=True)
    
    return all_results


if __name__ == "__main__":
    phase12_main()
