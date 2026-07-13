#!/usr/bin/env python
"""Compare two registered 3DGS renderers on identical scene/camera inputs."""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.nn.functional import conv2d


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from benchmark_framework import load_cameras_from_json, load_ply
from renderers import get_renderer


def compute_psnr(pred: torch.Tensor, reference: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - reference.float()) ** 2).item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, reference: torch.Tensor) -> float:
    """Windowed RGB SSIM for HWC images in [0, 1]."""
    pred = pred.float().permute(2, 0, 1).unsqueeze(0)
    reference = reference.float().permute(2, 0, 1).unsqueeze(0)
    coords = torch.arange(11, device=pred.device, dtype=pred.dtype) - 5
    kernel_1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 11, 11)

    def blur(image):
        return conv2d(image, kernel, padding=5, groups=3)

    mu_pred = blur(pred)
    mu_ref = blur(reference)
    var_pred = blur(pred * pred) - mu_pred * mu_pred
    var_ref = blur(reference * reference) - mu_ref * mu_ref
    covariance = blur(pred * reference) - mu_pred * mu_ref
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_pred * mu_ref + c1) * (2 * covariance + c2)) / (
        (mu_pred * mu_pred + mu_ref * mu_ref + c1)
        * (var_pred + var_ref + c2)
    )
    return score.mean().item()


def default_data_path(filename: str) -> str:
    for directory in (os.path.join(REPO_ROOT, "data"), os.path.join(SRC_DIR, "data")):
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return os.path.join(REPO_ROOT, "data", filename)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="gsplat_dense")
    parser.add_argument("--test", default="speedy_splat")
    parser.add_argument("--scene", default=default_data_path("scene.ply"))
    parser.add_argument("--cameras", default=default_data_path("cameras.json"))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--min-psnr", type=float, default=35.0)
    parser.add_argument("--min-ssim", type=float, default=0.95)
    parser.add_argument(
        "--output", default=os.path.join(REPO_ROOT, "results", "verified", "quality.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scene = load_ply(args.scene, device="cuda")
    cameras = load_cameras_from_json(args.cameras, device="cuda")
    reference = get_renderer(args.reference)
    test = get_renderer(args.test)
    if reference is None or test is None:
        raise SystemExit("Both renderers must be available")

    reference_scene = reference.prepare_scene(scene)
    test_scene = test.prepare_scene(scene)
    frame_results = []
    psnrs = []
    ssims = []
    with torch.inference_mode():
        for index, camera in enumerate(cameras[: args.frames]):
            reference_image = reference.render(reference_scene, camera)
            test_image = test.render(test_scene, camera)
            torch.cuda.synchronize()
            if reference_image.shape != test_image.shape:
                raise RuntimeError(
                    f"Shape mismatch: {reference_image.shape} vs {test_image.shape}"
                )
            psnr = compute_psnr(test_image, reference_image)
            ssim = compute_ssim(test_image, reference_image)
            max_error = (test_image - reference_image).abs().max().item()
            psnrs.append(psnr)
            ssims.append(ssim)
            frame_results.append(
                {
                    "frame": index,
                    "psnr": psnr if math.isfinite(psnr) else "inf",
                    "ssim": ssim,
                    "max_error": max_error,
                }
            )
            print(
                f"frame={index:03d}  PSNR={psnr:8.3f} dB  "
                f"SSIM={ssim:.6f}  max_error={max_error:.6f}"
            )

    passed = min(psnrs) >= args.min_psnr and min(ssims) >= args.min_ssim
    mean_psnr = float(np.mean(psnrs))
    min_psnr = float(np.min(psnrs))
    report = {
        "reference": args.reference,
        "test": args.test,
        "scene": args.scene,
        "frames": frame_results,
        "summary": {
            "mean_psnr": mean_psnr if math.isfinite(mean_psnr) else "inf",
            "min_psnr": min_psnr if math.isfinite(min_psnr) else "inf",
            "mean_ssim": float(np.mean(ssims)),
            "min_ssim": float(np.min(ssims)),
            "min_psnr_required": args.min_psnr,
            "min_ssim_required": args.min_ssim,
            "passed": passed,
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report["summary"], indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
