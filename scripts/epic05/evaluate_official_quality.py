#!/usr/bin/env python3
"""
EPIC-05 Official Dataset Quality Evaluation.

For each official scene and each tile size:
  1. Render all cameras
  2. Compute per-camera PSNR, SSIM, LPIPS against ground truth
  3. Compute pixel-level equivalence between tile16 and tile32 renderings
  4. Apply quality gates
  5. Generate per-camera + scene-aggregate quality report

Usage:
    python scripts/epic05/evaluate_official_quality.py \\
        --scene bicycle --resolution 1080p --tile-sizes 16 32

    python scripts/epic05/evaluate_official_quality.py \\
        --all --resolution 1080p --tile-sizes 8 16 32
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import conv2d, interpolate

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
from adapters.quality import QualityThresholds, evaluate_quality_gate

# ---------------------------------------------------------------------------
# GPU renderer
# ---------------------------------------------------------------------------
class GsplatTileRenderer:
    """gsplat rasterization wrapper with configurable tile_size."""

    def __init__(self, tile_size: int = 16, packed: bool = True,
                 device: str = "cuda"):
        self.tile_size = tile_size
        self.packed = packed
        self.device = device

    def is_available(self) -> bool:
        try:
            from gsplat import rasterization  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    def prepare_scene(self, scene_data: dict) -> dict:
        return {
            **scene_data,
            "quats": torch.nn.functional.normalize(
                scene_data["rotations"], dim=-1
            ).contiguous(),
            "scales_activated": torch.exp(scene_data["scales"]).contiguous(),
            "opacities_activated": torch.sigmoid(
                scene_data["opacity"]
            ).contiguous(),
            "shs": scene_data["shs"].contiguous(),
        }

    def render(self, scene_data: dict, camera) -> torch.Tensor:
        from gsplat import rasterization
        rendered, _, _ = rasterization(
            means=scene_data["xyz"],
            quats=scene_data["quats"],
            scales=scene_data["scales_activated"],
            opacities=scene_data["opacities_activated"],
            colors=scene_data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.image_width,
            height=camera.image_height,
            tile_size=self.tile_size,
            sh_degree=scene_data.get("sh_degree", 3),
            packed=self.packed,
            render_mode="RGB",
        )
        return rendered[0].clamp(0, 1)


# ---------------------------------------------------------------------------
# Quality metrics (matching validate_quality.py)
# ---------------------------------------------------------------------------
def compute_psnr(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """PSNR in dB, data range [0, 1]."""
    if pred.shape != ref.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {ref.shape}")
    mse = torch.mean((pred.float() - ref.float()) ** 2).item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """3DGS-compatible zero-padded RGB SSIM."""
    if min(pred.shape[:2]) < 11:
        raise ValueError("SSIM requires images >= 11x11")
    pred = pred.float().permute(2, 0, 1).unsqueeze(0)
    ref = ref.float().permute(2, 0, 1).unsqueeze(0)
    coords = torch.arange(11, device=pred.device, dtype=pred.dtype) - 5
    kernel_1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 11, 11)

    def blur(image):
        return conv2d(image, kernel, padding=5, groups=3)

    mu_pred, mu_ref = blur(pred), blur(ref)
    var_pred = blur(pred * pred) - mu_pred * mu_pred
    var_ref = blur(ref * ref) - mu_ref * mu_ref
    cov = blur(pred * ref) - mu_pred * mu_ref
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_pred * mu_ref + c1) * (2 * cov + c2)) / (
        (mu_pred * mu_pred + mu_ref * mu_ref + c1) * (var_pred + var_ref + c2)
    )
    return score.mean().item()


class LPIPSMetric:
    """Lightweight LPIPS wrapper (lazy init to avoid import burden)."""

    def __init__(self, device: str = "cuda", net: str = "vgg"):
        self.device = device
        self.net = net
        self._model = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "Install `lpips>=0.1.4` for LPIPS quality evaluation"
            ) from exc
        self._model = lpips.LPIPS(net=self.net).eval().to(self.device)

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor) -> float:
        self._lazy_init()
        pred_nchw = pred.permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        ref_nchw = ref.permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        with torch.inference_mode():
            return float(self._model(pred_nchw, ref_nchw).reshape(-1)[0].item())


# ---------------------------------------------------------------------------
# Pixel-level equivalence metrics
# ---------------------------------------------------------------------------
def compute_pixel_equivalence(img_a: torch.Tensor, img_b: torch.Tensor) -> dict:
    """Compute pixel-level differences between two rendered images."""
    diff = (img_a.float() - img_b.float()).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())
    rmse = float(torch.sqrt((diff ** 2).mean()).item())
    fraction_above_1e3 = float((diff > 1e-3).float().mean().item())
    fraction_above_1e2 = float((diff > 1e-2).float().mean().item())
    return {
        "max_abs_pixel_error": round(max_err, 8),
        "mean_abs_pixel_error": round(mean_err, 8),
        "rmse": round(rmse, 8),
        "fraction_pixels_above_1e-3": round(fraction_above_1e3, 8),
        "fraction_pixels_above_1e-2": round(fraction_above_1e2, 8),
        "exactly_identical": bool(max_err == 0.0),
    }


# ---------------------------------------------------------------------------
# Scene + resolution registry
# ---------------------------------------------------------------------------
RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

OFFICIAL_SCENES = {
    "bicycle": {
        "scene_id": "bicycle",
        "dataset_family": "Mip-NeRF 360",
        "scene_path": "data/official/mipnerf360/bicycle/point_cloud.ply",
        "camera_path": "data/official/mipnerf360/bicycle/cameras.json",
        "num_gaussians": 6131954,
        "camera_count": 281,
    },
    "garden": {
        "scene_id": "garden",
        "dataset_family": "Mip-NeRF 360",
        "scene_path": "data/official/mipnerf360/garden/point_cloud.ply",
        "camera_path": "data/official/mipnerf360/garden/cameras.json",
        "num_gaussians": 5834784,
        "camera_count": 250,
    },
    "room": {
        "scene_id": "room",
        "dataset_family": "Mip-NeRF 360",
        "scene_path": "data/official/mipnerf360/room/point_cloud.ply",
        "camera_path": "data/official/mipnerf360/room/cameras.json",
        "num_gaussians": 1593376,
        "camera_count": 217,
    },
}


# ---------------------------------------------------------------------------
# Quality evaluation functions
# ---------------------------------------------------------------------------
def _json_number(value: float):
    return value if math.isfinite(value) else "inf"


def evaluate_scene_quality(
    scene_id: str,
    scene_path: Path,
    camera_path: Path,
    tile_sizes: List[int],
    target_resolution: Tuple[int, int],
    resolution_label: str,
    gt_images_dir: Optional[Path] = None,
    num_cameras: Optional[int] = None,
    warmup_frames: int = 5,
    lpips_net: str = "vgg",
) -> Dict:
    """Run full quality evaluation for all tile sizes on one scene.

    For each tile_size:
      1. Render all cameras
      2. Compute PSNR/SSIM/LPIPS vs GT (if GT dir provided)
      3. Compute pixel-level equivalence between tile variants

    Returns a dict with per-tile quality data and cross-tile comparisons.
    """
    device = "cuda"

    print(f"\n{'='*70}")
    print(f"  Quality Evaluation: {scene_id}")
    print(f"  Tile sizes: {tile_sizes}")
    print(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    print(f"  GT images: {gt_images_dir}")
    print(f"{'='*70}")

    # Load scene
    print(f"\n  Loading scene...", end=" ", flush=True)
    scene_data = load_ply(str(scene_path), device=device)
    print(f"done ({scene_data['num_points']:,} gaussians)")

    # Load cameras
    print(f"  Loading cameras...", end=" ", flush=True)
    cameras = load_cameras_from_json(str(camera_path), device=device)
    cameras = resize_cameras(cameras, *target_resolution)
    print(f"loaded {len(cameras)} cameras, "
          f"{cameras[0].image_width}x{cameras[0].image_height}")

    if num_cameras is not None and num_cameras < len(cameras):
        cameras = cameras[:num_cameras]
        print(f"  Using first {num_cameras} cameras")

    # Load ground truth images
    gt_images = None
    if gt_images_dir and gt_images_dir.exists():
        gt_images = _load_ground_truth_images(
            cameras, gt_images_dir, device, background="black"
        )
        print(f"  Loaded {len(gt_images)} GT images from {gt_images_dir}")
    else:
        print(f"  No GT images available (dir: {gt_images_dir})")

    # Cold-start GPU
    print(f"  Cold-start preparation...", end=" ", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    dummy = GsplatTileRenderer(tile_size=tile_sizes[0], packed=True)
    dummy_data = dummy.prepare_scene(scene_data)
    with torch.no_grad():
        _ = dummy.render(dummy_data, cameras[0])
    torch.cuda.synchronize()
    del dummy, dummy_data
    gc.collect()
    torch.cuda.empty_cache()
    print("done")

    # LPIPS metric (lazy init)
    lpips_metric = LPIPSMetric(device=device, net=lpips_net)

    # Run rendering for each tile size
    tile_renders: Dict[int, List[torch.Tensor]] = {}
    tile_quality: Dict[int, Dict] = {}

    for ts in tile_sizes:
        print(f"\n  --- tile_size={ts} ---")
        renderer = GsplatTileRenderer(tile_size=ts, packed=True)
        prep_data = renderer.prepare_scene(scene_data)

        # Warmup
        for f in range(warmup_frames):
            cam = cameras[f % len(cameras)]
            with torch.no_grad():
                renderer.render(prep_data, cam)
        torch.cuda.synchronize()

        # Render all cameras
        rendered_images = []
        psnrs, ssims, lpips_values = [], [], []
        t0 = time.perf_counter()

        for idx, cam in enumerate(cameras):
            with torch.no_grad():
                prediction = renderer.render(prep_data, cam)
            torch.cuda.synchronize()
            rendered_images.append(prediction.cpu())

            # Compute quality against GT
            if gt_images is not None and idx < len(gt_images):
                gt = gt_images[idx].to(device)
                psnr_val = compute_psnr(prediction, gt)
                ssim_val = compute_ssim(prediction, gt)
                lpips_val = lpips_metric(prediction, gt)
                psnrs.append(psnr_val)
                ssims.append(ssim_val)
                lpips_values.append(lpips_val)

            if (idx + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    {idx+1}/{len(cameras)} ({elapsed/(idx+1)*1000:.0f}ms/frame)")

        total_time = time.perf_counter() - t0
        print(f"    Rendered {len(cameras)} frames in {total_time:.1f}s "
              f"({total_time/len(cameras)*1000:.1f}ms avg)")

        tile_renders[ts] = rendered_images

        # Quality summary
        quality_summary = {
            "num_views": len(psnrs) if psnrs else 0,
        }
        if psnrs:
            quality_summary["mean_psnr_db"] = _json_number(float(np.mean(psnrs)))
            quality_summary["min_psnr_db"] = _json_number(float(np.min(psnrs)))
            quality_summary["mean_ssim"] = float(np.mean(ssims))
            quality_summary["min_ssim"] = float(np.min(ssims))
            quality_summary["mean_lpips"] = float(np.mean(lpips_values))
            quality_summary["max_lpips"] = float(np.max(lpips_values))
            quality_summary["per_view"] = [
                {
                    "camera_index": i,
                    "image_name": cameras[i].image_name or str(i),
                    "psnr_db": _json_number(psnrs[i]),
                    "ssim": ssims[i],
                    "lpips": lpips_values[i],
                }
                for i in range(len(psnrs))
            ]
            print(f"    Quality vs GT ({len(psnrs)} views):")
            print(f"      PSNR:  {quality_summary['mean_psnr_db']:.3f} dB")
            print(f"      SSIM:  {quality_summary['mean_ssim']:.6f}")
            print(f"      LPIPS: {quality_summary['mean_lpips']:.6f}")
        else:
            quality_summary["quality_status"] = "unavailable"

        tile_quality[ts] = quality_summary

    # Cross-tile comparisons (pixel-level equivalence)
    cross_tile: Dict[str, Dict] = {}
    for i, ts_a in enumerate(tile_sizes):
        for ts_b in tile_sizes[i + 1:]:
            key = f"tile{ts_a}_vs_tile{ts_b}"
            print(f"\n  --- Pixel equivalence: tile{ts_a} vs tile{ts_b} ---")
            per_frame = []
            max_errors, mean_errors, rmses = [], [], []
            for idx in range(min(len(tile_renders[ts_a]), len(tile_renders[ts_b]))):
                img_a = tile_renders[ts_a][idx].to(device)
                img_b = tile_renders[ts_b][idx].to(device)
                eq = compute_pixel_equivalence(img_a, img_b)
                eq["camera_index"] = idx
                per_frame.append(eq)
                max_errors.append(eq["max_abs_pixel_error"])
                mean_errors.append(eq["mean_abs_pixel_error"])
                rmses.append(eq["rmse"])

            cross_tile[key] = {
                "aggregate": {
                    "max_max_abs_pixel_error": float(np.max(max_errors)),
                    "mean_max_abs_pixel_error": float(np.mean(max_errors)),
                    "mean_mean_abs_pixel_error": float(np.mean(mean_errors)),
                    "mean_rmse": float(np.mean(rmses)),
                    "max_rmse": float(np.max(rmses)),
                    "exactly_identical_all_views": all(
                        eq["exactly_identical"] for eq in per_frame
                    ),
                },
                "per_frame": per_frame,
            }
            print(f"      Max pixel error: {cross_tile[key]['aggregate']['max_max_abs_pixel_error']:.6f}")
            print(f"      Mean pixel error: {cross_tile[key]['aggregate']['mean_mean_abs_pixel_error']:.8f}")
            print(f"      Mean RMSE: {cross_tile[key]['aggregate']['mean_rmse']:.8f}")
            print(f"      Exactly identical: {cross_tile[key]['aggregate']['exactly_identical_all_views']}")

    # Apply quality gate for tile comparison
    quality_gate_results = {}
    if 16 in tile_quality and 32 in tile_quality:
        q16 = tile_quality[16]
        q32 = tile_quality[32]
        if "mean_psnr_db" in q16 and "mean_psnr_db" in q32:
            delta_psnr = q32["mean_psnr_db"] - q16["mean_psnr_db"]
            delta_ssim = q32["mean_ssim"] - q16["mean_ssim"]
            delta_lpips = q32["mean_lpips"] - q16["mean_lpips"]

            gate = evaluate_quality_gate(
                q32["mean_psnr_db"],
                q32["mean_ssim"],
                q32["mean_lpips"],
                QualityThresholds(
                    min_psnr_db=q16["mean_psnr_db"] - 0.10,  # allow 0.1dB drop
                    min_ssim=q16["mean_ssim"] - 0.003,  # allow 0.003 drop
                    max_lpips=q16["mean_lpips"] + 0.005,  # allow 0.005 increase
                ),
            )

            quality_gate_results["tile16_vs_tile32"] = {
                "delta_psnr_db": round(delta_psnr, 4),
                "delta_ssim": round(delta_ssim, 6),
                "delta_lpips": round(delta_lpips, 6),
                "quality_equivalent": bool(gate.passed),
                "gate_failures": list(gate.failures),
                "gate_status": "QUALITY_PASS" if gate.passed else "QUALITY_FAIL",
            }
            print(f"\n  --- Quality Gate: tile16 vs tile32 ---")
            print(f"      ΔPSNR:  {delta_psnr:.4f} dB")
            print(f"      ΔSSIM:  {delta_ssim:.6f}")
            print(f"      ΔLPIPS: {delta_lpips:.6f}")
            print(f"      Status: {quality_gate_results['tile16_vs_tile32']['gate_status']}")
        else:
            quality_gate_results["tile16_vs_tile32"] = {
                "status": "quality_gate_not_applicable",
                "reason": "GT quality data unavailable for one or both tile sizes",
            }

    result = {
        "scene_id": scene_id,
        "dataset_family": "Mip-NeRF 360",
        "scene_path": str(scene_path),
        "camera_path": str(camera_path),
        "num_gaussians": scene_data["num_points"],
        "resolution": list(target_resolution),
        "resolution_label": resolution_label,
        "num_cameras_evaluated": len(cameras),
        "gt_images_available": gt_images is not None,
        "gt_images_dir": str(gt_images_dir) if gt_images_dir else None,
        "tile_sizes": tile_sizes,
        "tile_quality": {f"tile{ts}": q for ts, q in tile_quality.items()},
        "cross_tile_equivalence": cross_tile,
        "quality_gate": quality_gate_results,
    }

    return result


def _load_ground_truth_images(
    cameras, gt_dir: Path, device: str, background: str = "black"
) -> List[torch.Tensor]:
    """Load ground truth images matching the camera image_names."""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required to load GT images")

    # Build image index
    extensions = {".png", ".jpg", ".jpeg"}
    index = {}
    for fpath in gt_dir.iterdir():
        if fpath.is_file() and fpath.suffix.lower() in extensions:
            index[fpath.stem] = fpath

    gt_images = []
    for cam in cameras:
        if not cam.image_name:
            continue
        stem = Path(cam.image_name).stem
        if stem not in index:
            continue
        fpath = index[stem]
        with Image.open(fpath) as source:
            rgba = np.asarray(source.convert("RGBA"), dtype=np.float32) / 255.0
        rgb, alpha = rgba[..., :3], rgba[..., 3:4]
        bg_val = 1.0 if background == "white" else 0.0
        rgb = rgb * alpha + bg_val * (1.0 - alpha)
        gt_tensor = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
        gt_images.append(gt_tensor)

    return gt_images


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scene", nargs="+",
        choices=list(OFFICIAL_SCENES.keys()),
        default=None,
    )
    p.add_argument("--all", action="store_true")
    p.add_argument(
        "--resolution", choices=list(RESOLUTION_PRESETS.keys()),
        default="1080p",
    )
    p.add_argument("--tile-sizes", nargs="+", type=int,
                    default=[8, 16, 32])
    p.add_argument("--gt-dir", type=Path, default=None,
                    help="Override ground truth images directory")
    p.add_argument("--num-cameras", type=int, default=None,
                    help="Limit number of cameras (for faster testing)")
    p.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"),
                    default="vgg")
    p.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results" / "epic05" / "official" / "quality",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.scene and not args.all:
        raise SystemExit("Specify --scene or --all")
    if args.all:
        scene_ids = list(OFFICIAL_SCENES.keys())
    else:
        scene_ids = args.scene

    target_resolution = RESOLUTION_PRESETS[args.resolution]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for scene_id in scene_ids:
        scene_info = OFFICIAL_SCENES[scene_id]
        scene_path = REPO_ROOT / scene_info["scene_path"]
        camera_path = REPO_ROOT / scene_info["camera_path"]

        if not scene_path.exists():
            print(f"ERROR: Scene not found: {scene_path}")
            continue
        if not camera_path.exists():
            print(f"ERROR: Camera path not found: {camera_path}")
            continue

        # GT directory: check Mip-NeRF 360 official dataset location
        gt_dir = args.gt_dir
        if gt_dir is None:
            # Try standard locations
            candidates = [
                REPO_ROOT / "data" / "datasets" / "mipnerf360" / scene_id / "images",
                REPO_ROOT / "data" / "official" / "mipnerf360" / scene_id / "input",
                REPO_ROOT / "data" / "datasets" / "mipnerf360" / scene_id,
            ]
            for candidate in candidates:
                if candidate.exists():
                    gt_dir = candidate
                    break

        result = evaluate_scene_quality(
            scene_id=scene_id,
            scene_path=scene_path,
            camera_path=camera_path,
            tile_sizes=args.tile_sizes,
            target_resolution=target_resolution,
            resolution_label=args.resolution,
            gt_images_dir=gt_dir,
            num_cameras=args.num_cameras,
            lpips_net=args.lpips_net,
        )

        # Save per-scene result
        out_path = output_dir / f"quality_{scene_id}_{args.resolution}_{timestamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"\n  Quality report saved: {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
