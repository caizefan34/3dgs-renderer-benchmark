#!/usr/bin/env python
"""Measure each 3DGS renderer against held-out ground-truth images.

The camera file may use this project's schema or the ``cameras.json`` list
exported by graphdeco-inria/gaussian-splatting. Every evaluated camera must
contain an ``image_name``/``img_name`` that resolves under --ground-truth-dir.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import conv2d


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from benchmark_framework import load_cameras_from_json, load_ply
from renderers import get_renderer


def _validate_images(pred: torch.Tensor, reference: torch.Tensor) -> None:
    if pred.shape != reference.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Expected matching HWC RGB images, got {pred.shape} and {reference.shape}")
    for name, image in (("prediction", pred), ("ground truth", reference)):
        if not torch.isfinite(image).all():
            raise ValueError(f"{name} contains non-finite values")
        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError(f"{name} is outside [0, 1]")


def compute_psnr(pred: torch.Tensor, reference: torch.Tensor) -> float:
    _validate_images(pred, reference)
    mse = torch.mean((pred.float() - reference.float()) ** 2).item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, reference: torch.Tensor) -> float:
    """Original-3DGS-compatible valid-window RGB SSIM for images in [0, 1]."""
    _validate_images(pred, reference)
    if min(pred.shape[:2]) < 11:
        raise ValueError("SSIM requires images at least 11x11 pixels")
    pred = pred.float().permute(2, 0, 1).unsqueeze(0)
    reference = reference.float().permute(2, 0, 1).unsqueeze(0)
    coords = torch.arange(11, device=pred.device, dtype=pred.dtype) - 5
    kernel_1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 11, 11)

    def blur(image):
        return conv2d(image, kernel, groups=3)

    mu_pred, mu_ref = blur(pred), blur(reference)
    var_pred = blur(pred * pred) - mu_pred * mu_pred
    var_ref = blur(reference * reference) - mu_ref * mu_ref
    covariance = blur(pred * reference) - mu_pred * mu_ref
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_pred * mu_ref + c1) * (2 * covariance + c2)) / (
        (mu_pred * mu_pred + mu_ref * mu_ref + c1) * (var_pred + var_ref + c2)
    )
    return score.mean().item()


class LPIPSMetric:
    def __init__(self, device: str = "cuda", net: str = "alex", model=None):
        if model is None:
            try:
                import lpips
            except ImportError as exc:
                raise RuntimeError(
                    "LPIPS is required for GT quality evaluation; install `lpips>=0.1.4`"
                ) from exc
            model = lpips.LPIPS(net=net)
        self.model = model.eval().to(device)

    def __call__(self, pred: torch.Tensor, reference: torch.Tensor) -> float:
        _validate_images(pred, reference)
        pred_nchw = pred.permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        ref_nchw = reference.permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        with torch.inference_mode():
            return float(self.model(pred_nchw, ref_nchw).reshape(-1)[0].item())


def _image_index(directory: str) -> dict:
    extensions = {".png", ".jpg", ".jpeg"}
    index = {}
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in index:
                raise ValueError(f"Duplicate ground-truth image stem: {path.stem}")
            index[path.stem] = path
    return index


def resolve_ground_truth(index: dict, image_name: str) -> Path:
    if not image_name:
        raise ValueError("Every quality camera needs image_name/img_name")
    stem = Path(image_name).stem
    if stem not in index:
        raise FileNotFoundError(f"No ground-truth image for camera {image_name!r}")
    return index[stem]


def load_ground_truth(path: Path, device: str, background: str) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load ground-truth images") from exc
    with Image.open(path) as source:
        rgba = np.asarray(source.convert("RGBA"), dtype=np.float32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3:4]
    background_value = 1.0 if background == "white" else 0.0
    rgb = rgb * alpha + background_value * (1.0 - alpha)
    return torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)


def camera_at_image_size(camera, image: torch.Tensor):
    """Scale intrinsics to a GT image resolution while preserving its field of view."""
    height, width = image.shape[:2]
    source_aspect = camera.image_width / camera.image_height
    target_aspect = width / height
    if abs(source_aspect - target_aspect) / source_aspect > 0.002:
        raise ValueError(
            f"GT aspect ratio {width}x{height} does not match camera "
            f"{camera.image_width}x{camera.image_height}"
        )
    K = camera.K.clone()
    K[0, 0], K[0, 2] = width / (2.0 * camera.tanfovx), width / 2.0
    K[1, 1], K[1, 2] = height / (2.0 * camera.tanfovy), height / 2.0
    return replace(camera, image_width=width, image_height=height, K=K)


def _json_number(value: float):
    return value if math.isfinite(value) else "inf"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(psnrs, ssims, lpips_values, args) -> dict:
    result = {
        "mean_psnr_db": _json_number(float(np.mean(psnrs))),
        "min_psnr_db": _json_number(float(np.min(psnrs))),
        "mean_ssim": float(np.mean(ssims)),
        "min_ssim": float(np.min(ssims)),
        "mean_lpips": float(np.mean(lpips_values)),
        "max_lpips": float(np.max(lpips_values)),
        "num_views": len(psnrs),
    }
    result["passed"] = (
        float(np.mean(psnrs)) >= args.min_psnr
        and float(np.mean(ssims)) >= args.min_ssim
        and float(np.mean(lpips_values)) <= args.max_lpips
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderers", nargs="+", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument(
        "--require-all-ground-truth",
        action="store_true",
        help="Fail if any camera lacks a matching GT image instead of selecting the held-out subset",
    )
    parser.add_argument(
        "--background", choices=("black",), default="black",
        help="Renderer and GT background (only black is implemented by all adapters)",
    )
    parser.add_argument(
        "--split-label", default="unspecified",
        help="Provenance label recorded in the report; the tool cannot infer train/test membership",
    )
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--min-psnr", type=float, default=0.0)
    parser.add_argument("--min-ssim", type=float, default=0.0)
    parser.add_argument("--max-lpips", type=float, default=1.0)
    parser.add_argument("--baseline-renderer", default=None)
    parser.add_argument("--max-psnr-drop", type=float, default=0.1)
    parser.add_argument("--max-ssim-drop", type=float, default=0.001)
    parser.add_argument("--max-lpips-increase", type=float, default=0.001)
    parser.add_argument(
        "--output", default=os.path.join(REPO_ROOT, "results", "verified", "quality_gt.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    scene = load_ply(args.scene, device=device)
    cameras = load_cameras_from_json(args.cameras, device=device)
    image_index = _image_index(args.ground_truth_dir)
    pairs, missing = [], []
    for camera in cameras:
        if camera.image_name and Path(camera.image_name).stem in image_index:
            pairs.append((camera, resolve_ground_truth(image_index, camera.image_name)))
        else:
            missing.append(camera.image_name)
    if args.require_all_ground_truth and missing:
        raise FileNotFoundError(f"{len(missing)} cameras have no matching GT image")
    pairs = pairs[: args.frames] if args.frames is not None else pairs
    if not pairs:
        raise SystemExit("No cameras matched ground-truth images")
    print(f"Matched {len(pairs)} GT views; skipped {len(missing)} other cameras")
    evaluation_pairs = []
    for camera, image_path in pairs:
        reference_cpu = load_ground_truth(image_path, "cpu", args.background)
        evaluation_pairs.append(
            (camera_at_image_size(camera, reference_cpu), image_path, reference_cpu)
        )
    lpips_metric = LPIPSMetric(device=device, net=args.lpips_net)
    renderer_results = []

    for renderer_name in args.renderers:
        renderer = get_renderer(renderer_name, device=device)
        if renderer is None:
            raise SystemExit(f"Renderer {renderer_name!r} is not available")
        prepared = renderer.prepare_scene(scene)
        frames, psnrs, ssims, lpips_values = [], [], [], []
        with torch.inference_mode():
            for index, (camera, image_path, reference_cpu) in enumerate(evaluation_pairs):
                reference = reference_cpu.to(device)
                prediction = renderer.render(prepared, camera)
                torch.cuda.synchronize()
                psnr = compute_psnr(prediction, reference)
                ssim = compute_ssim(prediction, reference)
                lpips_value = lpips_metric(prediction, reference)
                psnrs.append(psnr)
                ssims.append(ssim)
                lpips_values.append(lpips_value)
                frames.append({
                    "frame": index,
                    "image": image_path.name,
                    "psnr_db": _json_number(psnr),
                    "ssim": ssim,
                    "lpips": lpips_value,
                })
                print(
                    f"{renderer_name} frame={index:03d} PSNR={psnr:8.3f}dB "
                    f"SSIM={ssim:.6f} LPIPS={lpips_value:.6f}"
                )
        renderer_results.append({
            "renderer": renderer_name,
            "metadata": renderer.metadata(),
            "quality": _summary(psnrs, ssims, lpips_values, args),
            "frames": frames,
        })

    if args.baseline_renderer:
        by_name = {result["renderer"]: result for result in renderer_results}
        if args.baseline_renderer not in by_name:
            raise ValueError("--baseline-renderer must also be listed in --renderers")
        baseline = by_name[args.baseline_renderer]["quality"]
        for result in renderer_results:
            quality = result["quality"]
            comparison = {
                "baseline": args.baseline_renderer,
                "mean_psnr_delta_db": quality["mean_psnr_db"] - baseline["mean_psnr_db"],
                "mean_ssim_delta": quality["mean_ssim"] - baseline["mean_ssim"],
                "mean_lpips_delta": quality["mean_lpips"] - baseline["mean_lpips"],
            }
            comparison["quality_equivalent"] = (
                comparison["mean_psnr_delta_db"] >= -args.max_psnr_drop
                and comparison["mean_ssim_delta"] >= -args.max_ssim_drop
                and comparison["mean_lpips_delta"] <= args.max_lpips_increase
            )
            result["baseline_comparison"] = comparison

    with open(args.cameras, "rb") as file:
        camera_sha256 = hashlib.sha256(file.read()).hexdigest()
    report = {
        "schema_version": 2,
        "reference": {
            "type": "ground_truth",
            "camera_manifest": os.path.abspath(args.cameras),
            "camera_manifest_sha256": camera_sha256,
            "ground_truth_dir": os.path.abspath(args.ground_truth_dir),
            "background": args.background,
            "split_label": args.split_label,
            "color_space": "decoded sRGB values in [0, 1]",
        },
        "scene": os.path.abspath(args.scene),
        "scene_sha256": _sha256_file(args.scene),
        "metric_protocol": {
            "psnr_data_range": 1.0,
            "psnr_aggregation": "mean of per-view PSNR",
            "ssim_window": 11,
            "ssim_sigma": 1.5,
            "ssim_padding": "valid",
            "lpips_net": args.lpips_net,
            "lpips_input_range": "[-1, 1]",
        },
        "thresholds": {
            "min_mean_psnr_db": args.min_psnr,
            "min_mean_ssim": args.min_ssim,
            "max_mean_lpips": args.max_lpips,
            "max_psnr_drop_vs_baseline_db": args.max_psnr_drop,
            "max_ssim_drop_vs_baseline": args.max_ssim_drop,
            "max_lpips_increase_vs_baseline": args.max_lpips_increase,
        },
        "results": renderer_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Wrote {args.output}")
    passed = all(r["quality"]["passed"] for r in renderer_results)
    if args.baseline_renderer:
        passed = passed and all(
            r["baseline_comparison"]["quality_equivalent"] for r in renderer_results
        )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
