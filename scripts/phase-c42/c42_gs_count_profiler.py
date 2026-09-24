#!/usr/bin/env python3
"""
C42 Discrepancy Analysis: Gaussian-count-controlled profiler.

Measures SSIM%, backward%, total iteration time, and C42 speedup (scale=1.0 vs 0.75)
at controlled Gaussian counts: 500K, 1M, 5M, 10M, 20M.

This proves whether the discrepancy between old P1 (+33.6%) and new Track B (+6.4%)
is caused by Gaussian count differences (configuration mismatch in pruning schedule).
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss

DEVICE = "cuda"
N_TIMING_ITERS = 30
SCENE = "room"
TARGET_GS_COUNTS = [500_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000]
SCALES = [1.0, 0.75]

def cuda_timer(func, n_iters=30, warmup=5):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters

def create_model_with_n_gaussians(n_target, sfm_data):
    """Create a model with approximately n_target Gaussians by subsampling or duplicating."""
    n_sfm = sfm_data["xyz"].shape[0]

    if n_target <= n_sfm:
        # Subsample
        indices = torch.randperm(n_sfm)[:n_target]
        xyz = sfm_data["xyz"][indices]
        scales = sfm_data["scales"][indices]
        rotations = sfm_data["rotations"][indices]
        shs = sfm_data["shs"][indices]
    else:
        # Duplicate with small noise
        n_copies = (n_target + n_sfm - 1) // n_sfm
        xyz_parts, scales_parts, rot_parts, shs_parts = [], [], [], []
        for i in range(n_copies):
            noise = torch.randn_like(sfm_data["xyz"]) * 0.01
            xyz_parts.append(sfm_data["xyz"] + noise)
            scales_parts.append(sfm_data["scales"])
            rot_parts.append(sfm_data["rotations"])
            shs_parts.append(sfm_data["shs"])
        xyz = torch.cat(xyz_parts)[:n_target]
        scales = torch.cat(scales_parts)[:n_target]
        rotations = torch.cat(rot_parts)[:n_target]
        shs = torch.cat(shs_parts)[:n_target]

    n = xyz.shape[0]
    model = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device=DEVICE)
    model.init_from_sfm(
        xyz=xyz,
        opacity_logit=torch.logit(torch.full((n, 1), 0.1, device=DEVICE)),
        scales_log=scales,
        rotations_raw=rotations,
        shs=shs)
    model.set_sh_degree(3)
    model.eval()
    return model

def time_ssim_only(pred, gt, scale):
    """Time only the D-SSIM computation."""
    if scale < 1.0:
        pred_s = F.interpolate(pred.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode="area").squeeze(0).permute(1, 2, 0)
        gt_s = F.interpolate(gt.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode="area").squeeze(0).permute(1, 2, 0)
    else:
        pred_s, gt_s = pred, gt

    def do_ssim():
        return d_ssim_loss(pred_s, gt_s)

    return cuda_timer(do_ssim)

def profile_at_gs_count(model, dataset, cam_idx=0, n_gaussians=0):
    """Profile a single training iteration at a given Gaussian count."""
    cam = dataset.get_camera(cam_idx)
    gt = dataset.get_gt_image(cam_idx)
    W, H = cam.image_width, cam.image_height
    viewmats = cam.viewmatrix.unsqueeze(0)
    Ks = cam.K.unsqueeze(0)

    results = {"n_gaussians": n_gaussians, "scales": {}}

    for scale in SCALES:
        scale_result = {}

        # --- Time SSIM only ---
        with torch.no_grad():
            data = model.forward()
            pred_r, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=viewmats, Ks=Ks, width=W, height=H,
                tile_size=16, packed=False, sh_degree=3,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB")
            pred = pred_r[0].detach().clamp(0, 1)

        t_ssim = time_ssim_only(pred, gt, scale)
        scale_result["ssim_ms"] = t_ssim

        # --- Time forward (rasterization only, no grad) ---
        def do_fwd():
            with torch.no_grad():
                data_f = model.forward()
                r, a, l = rasterization(
                    means=data_f["xyz"], quats=data_f["rotations"], scales=data_f["scales"],
                    opacities=data_f["opacity"], colors=data_f["shs"],
                    viewmats=viewmats, Ks=Ks, width=W, height=H,
                    tile_size=16, packed=False, sh_degree=3,
                    radius_clip=0.0, eps2d=0.1, render_mode="RGB")
                return r, a

        t_fwd = cuda_timer(do_fwd)
        scale_result["fwd_ms"] = t_fwd

        # --- Time full iteration (fwd + loss + bwd) ---
        def do_full_iter():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            data_f = model.forward()
            r, a, l = rasterization(
                means=data_f["xyz"], quats=data_f["rotations"], scales=data_f["scales"],
                opacities=data_f["opacity"], colors=data_f["shs"],
                viewmats=viewmats, Ks=Ks, width=W, height=H,
                tile_size=16, packed=False, sh_degree=3,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB")
            # Compute loss with the given scale
            l1 = F.l1_loss(r, gt.unsqueeze(0))
            if scale < 1.0:
                pred_s = F.interpolate(r[0].unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode="area").squeeze(0).permute(1, 2, 0)
                gt_s = F.interpolate(gt.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode="area").squeeze(0).permute(1, 2, 0)
            else:
                pred_s, gt_s = r[0], gt
            dsim = d_ssim_loss(pred_s, gt_s)
            loss = (1 - 0.2) * l1 + 0.2 * dsim
            loss.backward()

        t_total = cuda_timer(do_full_iter, n_iters=20, warmup=5)
        t_bwd = t_total - t_fwd - t_ssim  # approximate backward time

        scale_result["total_ms"] = t_total
        scale_result["bwd_ms"] = t_bwd
        scale_result["ssim_pct"] = t_ssim / t_total * 100
        scale_result["fwd_pct"] = t_fwd / t_total * 100
        scale_result["bwd_pct"] = t_bwd / t_total * 100

        print(f"    scale={scale}: SSIM={t_ssim:.2f}ms ({scale_result['ssim_pct']:.1f}%)  "
              f"Fwd={t_fwd:.2f}ms ({scale_result['fwd_pct']:.1f}%)  "
              f"Bwd={t_bwd:.2f}ms ({scale_result['bwd_pct']:.1f}%)  "
              f"Total={t_total:.2f}ms")

        results["scales"][str(scale)] = scale_result

    # Compute C42 speedup
    t_base = results["scales"]["1.0"]["total_ms"]
    t_scaled = results["scales"]["0.75"]["total_ms"]
    results["c42_speedup_pct"] = (t_base - t_scaled) / t_base * 100
    results["ssim_savings_ms"] = results["scales"]["1.0"]["ssim_ms"] - results["scales"]["0.75"]["ssim_ms"]
    results["ssim_savings_pct_of_total"] = results["ssim_savings_ms"] / t_base * 100

    print(f"    → C42 speedup: {results['c42_speedup_pct']:+.1f}%  "
          f"(SSIM savings: {results['ssim_savings_ms']:.2f}ms = {results['ssim_savings_pct_of_total']:.1f}% of total)")

    return results

def main():
    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(42)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    import gsplat
    print(f"gsplat: {gsplat.__version__}")

    dataset = GTDataset(scene=SCENE, repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint(SCENE, repo_root, device=DEVICE)
    print(f"SfM points: {sfm_data['xyz'].shape[0]:,}")

    all_results = {
        "gpu": gpu_name,
        "scene": SCENE,
        "config": {
            "n_timing_iters": N_TIMING_ITERS,
            "target_gs_counts": TARGET_GS_COUNTS,
            "scales_tested": SCALES,
            "camera_idx": 0,
        },
        "data_points": [],
    }

    for n_target in TARGET_GS_COUNTS:
        print(f"\n{'='*60}")
        print(f"Target GS count: {n_target:,}")
        print(f"{'='*60}")

        model = create_model_with_n_gaussians(n_target, sfm_data)
        actual_n = model.xyz.shape[0]
        print(f"  Actual GS: {actual_n:,}")

        result = profile_at_gs_count(model, dataset, cam_idx=0, n_gaussians=actual_n)
        all_results["data_points"].append(result)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: C42 speedup vs Gaussian count")
    print(f"{'='*60}")
    print(f"{'N_Gaussians':>12} | {'SSIM%(1.0)':>10} | {'Bwd%(1.0)':>10} | {'Total(1.0)':>10} | {'Total(0.75)':>11} | {'C42 speedup':>12}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*11}-+-{'-'*12}")
    for dp in all_results["data_points"]:
        s10 = dp["scales"]["1.0"]
        s075 = dp["scales"]["0.75"]
        print(f"{dp['n_gaussians']:>12,} | {s10['ssim_pct']:>9.1f}% | {s10['bwd_pct']:>9.1f}% | {s10['total_ms']:>9.1f}ms | {s075['total_ms']:>10.1f}ms | {dp['c42_speedup_pct']:>+11.1f}%")

    # Save
    save_path = repo_root / "results" / "a100" / "phase-c42" / "c42_gs_count_profile.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {save_path}")

if __name__ == "__main__":
    main()
