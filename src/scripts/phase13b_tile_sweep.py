#!/usr/bin/env python
"""
Phase 13B — Expanded Tile-Size Sweep + Ground-Truth Quality Audit

Tests tile_sizes: 4, 8, 12, 16, 20, 24, 28, 32
Across scenes: room, bicycle, garden
With: forward, backward, forward+backward timing, workload stats, GT quality
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_framework import (
    load_ply,
    load_cameras_from_json,
    Camera,
    resize_cameras,
)

# Try torchmetrics
try:
    from torchmetrics.image.psnr import PeakSignalNoiseRatio
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("WARNING: torchmetrics not available.")


TILE_SIZES = [4, 8, 12, 16, 20, 24, 28, 32]
SCENES = ["room", "bicycle", "garden"]
RES_W, RES_H = 1920, 1080


def load_scene(scene_name: str, device: str = "cuda"):
    """Load SfM point cloud and cameras for a scene."""
    scene_dir = PROJECT_ROOT / "data" / "official" / "mipnerf360" / scene_name
    ply_path = scene_dir / "point_cloud.ply"
    cam_path = scene_dir / "cameras.json"

    if not ply_path.exists():
        raise FileNotFoundError(f"Scene not found: {ply_path}")
    if not cam_path.exists():
        raise FileNotFoundError(f"Cameras not found: {cam_path}")

    scene_data = load_ply(str(ply_path), device=device)
    cameras = load_cameras_from_json(str(cam_path), device=device)
    cameras = resize_cameras(cameras, RES_W, RES_H)

    print(f"Loaded {scene_name}: {scene_data['num_points']} Gaussians, {len(cameras)} cameras @ {RES_W}x{RES_H}")
    return scene_data, cameras


def prepare_scene_data(scene_data: dict, device: str = "cuda") -> dict:
    """Prepare scene tensors for rasterization."""
    return {
        "xyz": scene_data["xyz"].to(device, non_blocking=True).contiguous(),
        "rotations": scene_data["rotations"].to(device, non_blocking=True).contiguous(),
        "scales": scene_data["scales"].to(device, non_blocking=True).contiguous(),
        "opacity": scene_data["opacity"].to(device, non_blocking=True).contiguous(),
        "shs": scene_data["shs"].to(device, non_blocking=True).contiguous(),
        "sh_degree": scene_data.get("sh_degree", 3),
    }


def get_gt_image(scene_name: str, camera_idx: int = 0) -> Optional[torch.Tensor]:
    """Load a GT image for quality comparison."""
    gt_dir = PROJECT_ROOT / "data" / "datasets" / "mipnerf360" / scene_name / "images"
    if not gt_dir.exists():
        print(f"WARNING: GT images not found at {gt_dir}")
        return None
    
    # Try to load the first image (camera order: first is train set)
    from PIL import Image
    import torchvision.transforms as T
    
    img_files = sorted(gt_dir.glob("*.png")) + sorted(gt_dir.glob("*.jpg"))
    if camera_idx >= len(img_files):
        print(f"WARNING: Only {len(img_files)} GT images, can't access index {camera_idx}")
        return None
    
    img = Image.open(img_files[camera_idx]).convert("RGB")
    img = img.resize((RES_W, RES_H), Image.LANCZOS)
    gt = T.ToTensor()(img).to("cuda").unsqueeze(0)  # [1, 3, H, W]
    return gt


def compute_quality(rendered: torch.Tensor, gt: torch.Tensor) -> dict:
    """Compute PSNR, SSIM, LPIPS between rendered and GT images.
    
    Args:
        rendered: [1, H, W, 3] tensor in [0,1] range
        gt: [1, 3, H, W] tensor in [0,1] range
    Returns:
        dict with psnr, ssim, lpips values
    """
    results = {}
    
    # Convert rendered to [1, 3, H, W]
    if rendered.dim() == 4 and rendered.shape[-1] == 3:
        rendered_chw = rendered.permute(0, 3, 1, 2)
    else:
        rendered_chw = rendered
    
    # PSNR
    mse = torch.mean((rendered_chw - gt) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    results["psnr"] = psnr.item()
    
    # SSIM (simple version)
    mu1 = torch.mean(rendered_chw, dim=[2, 3])
    mu2 = torch.mean(gt, dim=[2, 3])
    sigma1_sq = torch.var(rendered_chw, dim=[2, 3])
    sigma2_sq = torch.var(gt, dim=[2, 3])
    sigma12 = torch.mean((rendered_chw - mu1[:, :, None, None]) * (gt - mu2[:, :, None, None]), dim=[2, 3])
    
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    ssim_per_channel = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = ssim_per_channel.mean().item()
    results["ssim"] = ssim
    
    if TORCHMETRICS_AVAILABLE:
        try:
            lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
            lpips_val = lpips_fn(rendered_chw.clamp(0, 1), gt.clamp(0, 1))
            results["lpips"] = lpips_val.item()
        except Exception as e:
            print(f"  LPIPS failed: {e}")
            results["lpips"] = None
    else:
        results["lpips"] = None
    
    return results


def compute_pixel_equivalence(img1: torch.Tensor, img2: torch.Tensor) -> dict:
    """Compare two rendered images for pixel-level differences."""
    diff = (img1.float() - img2.float()).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "changed_pixel_ratio": (diff > 1e-5).float().mean().item(),
    }


def run_tile_benchmark(scene_data: dict, camera: Camera, tile_size: int,
                       device: str = "cuda", do_bwd: bool = True,
                       n_warmup: int = 3, n_batch: int = 5, n_repeat: int = 3) -> dict:
    """Run forward/backward benchmark for a single tile_size on one camera.
    
    Returns dict with timing, workload statistics, and rendered image.
    """
    from gsplat import rasterization

    sd = prepare_scene_data(scene_data, device)
    
    # Prepare camera tensors
    vm = camera.viewmatrix.unsqueeze(0).to(device)
    K = camera.K.unsqueeze(0).to(device)
    
    # Set requires_grad for backward
    means = sd["xyz"].requires_grad_(True)
    quats = torch.nn.functional.normalize(sd["rotations"], dim=-1).requires_grad_(True)
    scales_raw = sd["scales"]
    scales_activated = torch.exp(scales_raw).requires_grad_(True)
    opacities_activated = torch.sigmoid(sd["opacity"]).requires_grad_(True)
    shs = sd["shs"].requires_grad_(True)
    
    # Synthetic loss target for backward
    target = torch.zeros(1, RES_H, RES_W, 3, device=device)
    
    # Meta dict for results
    result = {"tile_size": tile_size, "status": "OK"}
    
    # Workload snapshot from first forward pass
    try:
        rendered_colors, rendered_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales_activated,
            opacities=opacities_activated,
            colors=shs,
            viewmats=vm,
            Ks=K,
            width=RES_W,
            height=RES_H,
            sh_degree=3,
            packed=True,
            tile_size=tile_size,
            render_mode="RGB",
        )
    except Exception as e:
        result["status"] = f"FAILED_FWD: {str(e)}"
        return result
    
    # Collect workload statistics from meta
    tile_width = meta["tile_width"]
    tile_height = meta["tile_height"]
    n_tiles = tile_width * tile_height
    
    # Count visible Gaussians
    radii = meta["radii"]
    visible_mask = (radii > 0).all(dim=-1)
    n_visible = visible_mask.sum().item()
    
    # Total intersections
    n_isects = meta.get("isect_ids", torch.tensor([0])).shape[0]
    
    # Intersections per tile from tile_offsets
    # tile_offsets shape: [..., tile_height, tile_width] - prefix-sum offsets
    isect_offsets = meta["isect_offsets"]
    # Flatten to 1D: [..., tile_height * tile_width]
    isect_offsets_1d = isect_offsets.reshape(-1, n_tiles)
    # Use first batch (image 0) offsets
    batch_offsets = isect_offsets_1d[0]  # [n_tiles]
    
    # Compute counts per tile from consecutive offsets
    if n_isects > 0:
        counts_list = []
        for t in range(n_tiles - 1):
            counts_list.append(batch_offsets[t + 1].item() - batch_offsets[t].item())
        counts_list.append(n_isects - batch_offsets[-1].item())
        isect_counts = torch.tensor(counts_list, device=device, dtype=torch.float)
    else:
        isect_counts = torch.zeros(n_tiles, device=device)
    
    # Sort counts for detailed statistics
    counts_sorted = isect_counts.sort().values
    n_nonzero = (counts_sorted > 0).sum().item()
    empty_ratio = 1.0 - n_nonzero / max(n_tiles, 1)
    
    non_empty = counts_sorted[counts_sorted > 0]
    if len(non_empty) > 0:
        mean_intersections = non_empty.mean().item()
        median_intersections = non_empty.median().item()
        p95_idx = min(int(len(non_empty) * 0.95), len(non_empty) - 1)
        p99_idx = min(int(len(non_empty) * 0.99), len(non_empty) - 1)
        p95 = non_empty[p95_idx].item()
        p99 = non_empty[p99_idx].item()
        max_intersections = non_empty[-1].item()
    else:
        mean_intersections = 0
        median_intersections = 0
        p95 = 0
        p99 = 0
        max_intersections = 0
    
    # TPG (intersections per Gaussian) statistics
    tiles_per_gauss = meta.get("tiles_per_gauss", None)
    if tiles_per_gauss is not None and tiles_per_gauss.numel() > 0:
        # tiles_per_gauss is [nnz] or [..., N] - only visible ones have >0
        tpg_raw = tiles_per_gauss
        if tpg_raw.dim() > 1:
            tpg_raw = tpg_raw.flatten()
        visible_tpg = tpg_raw[tpg_raw > 0].float()
        if len(visible_tpg) > 0:
            tpg_mean = visible_tpg.mean().item()
            tpg_std = visible_tpg.std().item()
            tpg_median = visible_tpg.median().item()
            tpg_sorted = visible_tpg.sort()[0]
            tpg_p95_idx = min(int(len(tpg_sorted) * 0.95), len(tpg_sorted) - 1)
            tpg_p99_idx = min(int(len(tpg_sorted) * 0.99), len(tpg_sorted) - 1)
            tpg_p95 = tpg_sorted[tpg_p95_idx].item()
            tpg_p99 = tpg_sorted[tpg_p99_idx].item()
            tpg_max = tpg_sorted[-1].item()
        else:
            tpg_mean = tpg_std = tpg_median = 0
            tpg_p95 = tpg_p99 = tpg_max = 0
    else:
        tpg_mean = tpg_std = tpg_median = 0
        tpg_p95 = tpg_p99 = tpg_max = 0
    
    # Geometry info
    result["geometry"] = {
        "image_width": RES_W,
        "image_height": RES_H,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "total_tiles": n_tiles,
        "pixels_per_tile": tile_size * tile_size,
        "threads_per_block": tile_size * tile_size,
    }
    
    # Gaussian/visibility info
    total_gaussians = scene_data["num_points"]
    radii_flat = radii.float()
    result["gaussian_stats"] = {
        "total_gaussians": total_gaussians,
        "visible_gaussians": int(n_visible),
        "radius_stats": {
            "radii_min": radii_flat.min().item() if radii_flat.numel() > 0 else 0,
            "radii_max": radii_flat.max().item() if radii_flat.numel() > 0 else 0,
            "radii_mean": radii_flat.mean().item() if radii_flat.numel() > 0 else 0,
        },
    }
    
    # Intersection workload
    result["intersection_workload"] = {
        "total_intersections": int(n_isects),
        "mean_intersections_per_tile": mean_intersections,
        "median_intersections_per_tile": median_intersections,
        "p95_intersections_per_tile": p95,
        "p99_intersections_per_tile": p99,
        "max_intersections_per_tile": max_intersections,
        "empty_tile_ratio": empty_ratio,
        "tpg_mean": tpg_mean,
        "tpg_std": tpg_std,
        "tpg_median": tpg_median,
        "tpg_p95": tpg_p95,
        "tpg_p99": tpg_p99,
        "tpg_max": tpg_max,
    }
    
    # Timing benchmarks
    # Forward timing
    fwd_times = []
    for _ in range(n_warmup):
        rendered_colors, _, _ = rasterization(
            means=means, quats=quats, scales=scales_activated,
            opacities=opacities_activated, colors=shs,
            viewmats=vm, Ks=K, width=RES_W, height=RES_H,
            sh_degree=3, packed=True, tile_size=tile_size, render_mode="RGB",
        )
    
    for _ in range(n_repeat):
        for _ in range(n_batch):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            rendered_colors, _, _ = rasterization(
                means=means, quats=quats, scales=scales_activated,
                opacities=opacities_activated, colors=shs,
                viewmats=vm, Ks=K, width=RES_W, height=RES_H,
                sh_degree=3, packed=True, tile_size=tile_size, render_mode="RGB",
            )
            end.record()
            end.synchronize()
            fwd_times.append(start.elapsed_time(end))
        torch.cuda.empty_cache()
    
    fwd_times = np.array(fwd_times)
    result["forward_timing"] = {
        "mean_ms": float(fwd_times.mean()),
        "median_ms": float(np.median(fwd_times)),
        "std_ms": float(fwd_times.std()),
        "cv": float(fwd_times.std() / fwd_times.mean()) if fwd_times.mean() > 0 else 0,
        "min_ms": float(fwd_times.min()),
        "max_ms": float(fwd_times.max()),
        "n_samples": int(len(fwd_times)),
    }
    
    # Get rendered image for quality comparison
    rendered_image = rendered_colors[0].detach()  # [H, W, 3]
    
    # Forward+Backward timing
    if do_bwd and tile_size <= 32:
        fb_times = []
        for _ in range(n_warmup):
            rendered_colors, _, _ = rasterization(
                means=means, quats=quats, scales=scales_activated,
                opacities=opacities_activated, colors=shs,
                viewmats=vm, Ks=K, width=RES_W, height=RES_H,
                sh_degree=3, packed=True, tile_size=tile_size, render_mode="RGB",
            )
            loss = torch.nn.functional.mse_loss(rendered_colors, target)
            loss.backward()
            # Zero grads
            for p in [means, quats, scales_activated, opacities_activated, shs]:
                if p.grad is not None:
                    p.grad = None
        
        for _ in range(min(n_repeat, 2)):
            batch_fb = []
            for _ in range(min(n_batch, 3)):  # Fewer for FB
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                rendered_colors, _, _ = rasterization(
                    means=means, quats=quats, scales=scales_activated,
                    opacities=opacities_activated, colors=shs,
                    viewmats=vm, Ks=K, width=RES_W, height=RES_H,
                    sh_degree=3, packed=True, tile_size=tile_size, render_mode="RGB",
                )
                loss = torch.nn.functional.mse_loss(rendered_colors, target)
                loss.backward()
                end.record()
                end.synchronize()
                batch_fb.append(start.elapsed_time(end))
                # Zero grads
                for p in [means, quats, scales_activated, opacities_activated, shs]:
                    if p.grad is not None:
                        p.grad = None
            fb_times.extend(batch_fb)
            torch.cuda.empty_cache()
        
        fb_times = np.array(fb_times)
        bwd_times = fb_times - np.median(fwd_times)
        
        result["backward_timing"] = {
            "mean_ms": float(bwd_times.mean()),
            "median_ms": float(np.median(bwd_times)),
            "std_ms": float(bwd_times.std()),
            "cv": float(bwd_times.std() / bwd_times.mean()) if bwd_times.mean() > 0 else 0,
            "min_ms": float(bwd_times.min()),
            "max_ms": float(bwd_times.max()),
            "n_samples": int(len(bwd_times)),
            "inferred": True,
        }
        
        result["forward_backward_timing"] = {
            "mean_ms": float(fb_times.mean()),
            "median_ms": float(np.median(fb_times)),
            "std_ms": float(fb_times.std()),
            "cv": float(fb_times.std() / fb_times.mean()) if fb_times.mean() > 0 else 0,
            "min_ms": float(fb_times.min()),
            "max_ms": float(fb_times.max()),
            "n_samples": int(len(fb_times)),
        }
    
    result["rendered_image"] = rendered_image.cpu()  # Store for quality comparison
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 13B Tile-Size Sweep")
    parser.add_argument("--scene", choices=SCENES + ["all"], default="all")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=TILE_SIZES)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "epic05" / "phase13b"))
    parser.add_argument("--camera-idx", type=int, default=0, help="Camera index to use (first real camera)")
    parser.add_argument("--no-bwd", action="store_true", help="Skip backward timing")
    parser.add_argument("--gpu-device", type=str, default="cuda")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tile_sizes = [ts for ts in args.tile_sizes if ts in TILE_SIZES]
    scenes = SCENES if args.scene == "all" else [args.scene]
    device = args.gpu_device
    
    print(f"Phase 13B Tile-Size Expanded Sweep")
    print(f"  Scenes: {scenes}")
    print(f"  Tile sizes: {tile_sizes}")
    print(f"  Resolution: {RES_W}x{RES_H}")
    print(f"  Backward: {'YES' if not args.no_bwd else 'NO'}")
    print()
    
    all_results = {}
    
    for scene_name in scenes:
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}")
        print(f"{'='*60}")
        
        # Load scene
        scene_data, cameras = load_scene(scene_name, device)
        camera = cameras[min(args.camera_idx, len(cameras) - 1)]
        K = camera.K
        if isinstance(K, torch.Tensor):
            K_np = K.cpu().numpy() if K.is_cuda else K.numpy()
            fx, fy = K_np[0, 0], K_np[1, 1]
            cx, cy = K_np[0, 2], K_np[1, 2]
        else:
            fx = fy = cx = cy = 0.0
        print(f"  Camera {args.camera_idx}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        # Load GT image for quality
        gt_image = get_gt_image(scene_name, args.camera_idx)
        if gt_image is not None:
            print(f"  GT image loaded: {gt_image.shape}")
        else:
            print(f"  WARNING: No GT image available")
        
        scene_results = {}
        
        for ts in tile_sizes:
            print(f"\n  --- tile_size={ts} ---")
            
            try:
                result = run_tile_benchmark(
                    scene_data, camera, ts, device=device,
                    do_bwd=not args.no_bwd and ts <= 32,
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                result = {"tile_size": ts, "status": f"FAILED: {str(e)}"}
            
            if result["status"] == "OK":
                geo = result["geometry"]
                gs = result["gaussian_stats"]
                iw = result["intersection_workload"]
                ft = result["forward_timing"]
                print(f"    Geometry: {geo['tile_width']}x{geo['tile_height']} tiles, {geo['total_tiles']} total")
                print(f"    Gaussians: {gs['visible_gaussians']}/{gs['total_gaussians']} visible")
                print(f"    Intersections: {iw['total_intersections']} total, mean={iw['mean_intersections_per_tile']:.1f}/tile")
                print(f"    TPG: mean={iw['tpg_mean']:.2f}, std={iw['tpg_std']:.2f}, empty={iw['empty_tile_ratio']:.3f}")
                print(f"    Forward: median={ft['median_ms']:.3f}ms, mean={ft['mean_ms']:.3f}ms, cv={ft['cv']:.3f}")
                if "forward_backward_timing" in result:
                    fbt = result["forward_backward_timing"]
                    print(f"    Fwd+Bwd: median={fbt['median_ms']:.3f}ms")
                
                # Compute quality
                if gt_image is not None and "rendered_image" in result:
                    rendered = result["rendered_image"].to(device)
                    quality = compute_quality(rendered.unsqueeze(0), gt_image)
                    result["quality_vs_gt"] = quality
                    print(f"    Quality vs GT: PSNR={quality['psnr']:.2f}, SSIM={quality['ssim']:.4f}, LPIPS={quality.get('lpips', 'N/A')}")
                
                # Compare with tile_size=16 as baseline
                if ts != 16 and "rendered_image" in result:
                    # We'll compare with tile16 result later
                    pass
                
                # Clean up rendered image for storage (too large)
                if "rendered_image" in result:
                    del result["rendered_image"]
            else:
                print(f"    Status: {result['status']}")
            
            scene_results[str(ts)] = result
            torch.cuda.empty_cache()
        
        all_results[scene_name] = scene_results
        
        # Compare all tile sizes against tile_size=16 for pixel equivalence
        if "16" in scene_results and scene_results["16"]["status"] == "OK":
            print(f"\n  --- Pixel Equivalence vs tile_size=16 ---")
            # Re-render with tile16 for comparison
            sd = prepare_scene_data(scene_data, device)
            means = sd["xyz"]
            quats = torch.nn.functional.normalize(sd["rotations"], dim=-1).requires_grad_(False)
            scales_activated = torch.exp(sd["scales"]).requires_grad_(False)
            opacities_activated = torch.sigmoid(sd["opacity"]).requires_grad_(False)
            shs = sd["shs"].requires_grad_(False)
            
            tile16_img = None
            try:
                from gsplat import rasterization
                r16, _, _ = rasterization(
                    means=means, quats=quats, scales=scales_activated,
                    opacities=opacities_activated, colors=shs,
                    viewmats=camera.viewmatrix.unsqueeze(0).to(device),
                    Ks=camera.K.unsqueeze(0).to(device),
                    width=RES_W, height=RES_H, sh_degree=3, packed=True, tile_size=16,
                )
                tile16_img = r16[0].detach()
            except Exception as e:
                print(f"    tile16 re-render failed: {e}")
            
            if tile16_img is not None:
                for ts_str, ts_result in scene_results.items():
                    ts = int(ts_str)
                    if ts == 16 or ts_result["status"] != "OK":
                        continue
                    # Re-render at this tile_size
                    try:
                        r_ts, _, _ = rasterization(
                            means=means, quats=quats, scales=scales_activated,
                            opacities=opacities_activated, colors=shs,
                            viewmats=camera.viewmatrix.unsqueeze(0).to(device),
                            Ks=camera.K.unsqueeze(0).to(device),
                            width=RES_W, height=RES_H, sh_degree=3, packed=True, tile_size=ts,
                        )
                        ts_img = r_ts[0].detach()
                        eq = compute_pixel_equivalence(ts_img, tile16_img)
                        ts_result["pixel_equivalence_vs_tile16"] = eq
                    except Exception as e:
                        ts_result["pixel_equivalence_vs_tile16"] = {"status": f"FAILED: {e}"}
                    
                    if "pixel_equivalence_vs_tile16" in ts_result and "status" not in ts_result["pixel_equivalence_vs_tile16"]:
                        eq = ts_result["pixel_equivalence_vs_tile16"]
                        print(f"    tile{ts} vs tile16: max_diff={eq['max_abs_diff']:.6f}, mean_diff={eq['mean_abs_diff']:.6e}, changed={eq['changed_pixel_ratio']:.6f}")
        
        torch.cuda.empty_cache()
    
    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    
    # Strip rendered images from results for JSON serialization
    json_results = json.loads(json.dumps(all_results, default=str))
    
    output_file = output_dir / f"expanded_tile_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Also save as the canonical file
    canonical_file = output_dir / "expanded_tile_results.json"
    with open(canonical_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Canonical results saved to {canonical_file}")
    
    print(f"\nDone.")


if __name__ == "__main__":
    main()
