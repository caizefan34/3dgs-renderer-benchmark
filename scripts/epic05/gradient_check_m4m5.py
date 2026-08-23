#!/usr/bin/env python3
"""
EPIC-06 P1: Gradient Correctness for M4 (radius_clip) and M5 (eps2d).

Usage:
    python scripts/epic05/gradient_check_m4m5.py --module M4
    python scripts/epic05/gradient_check_m4m5.py --module M5
    python scripts/epic05/gradient_check_m4m5.py --all
"""
import argparse
import gc
import json
import os
import sys
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.cpp_extension as cpp_ext
cpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')
_msvc_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
for _p in [_msvc_dir, _cuda_bin]:
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gsplat import rasterization


def make_default_camera(w=1920, h=1080, device="cuda"):
    import numpy as np
    fx, fy = w * 0.6, h * 0.6
    tan_fov_x, tan_fov_y = w / (2.0 * fx), h / (2.0 * fy)
    near, far = 0.01, 100.0
    cam_to_world = np.eye(4, dtype=np.float32)
    cam_to_world[2, 3] = 5.0
    viewmat = torch.tensor(np.linalg.inv(cam_to_world).astype(np.float32), dtype=torch.float32, device=device)
    K = torch.tensor([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=torch.float32, device=device)
    return viewmat, K


def generate_scene(N=50000, device="cuda", seed=42):
    torch.manual_seed(seed)
    return {
        "xyz": torch.randn(N, 3, device=device) * 2.0,
        "rotations": F.normalize(torch.randn(N, 4, device=device), dim=-1),
        "scales": torch.log(torch.rand(N, 3, device=device) * 0.1 + 0.01),
        "opacity": torch.logit(torch.rand(N, 1, device=device) * 0.5 + 0.25),
        "shs": torch.randn(N, 16, 3, device=device) * 0.1,
        "num_points": N,
    }


def test_radius_clip(scene_data, device="cuda"):
    """Test gradient correctness across different radius_clip values."""
    viewmat, K = make_default_camera(device=device)
    w, h = 1920, 1080
    gt_image = torch.rand(h, w, 3, device=device)

    radius_clip_values = [0.0, 0.001, 0.005, 0.01]

    results = {}
    for rc in radius_clip_values:
        print(f"\n  === radius_clip={rc} ===")
        rc_results = {}

        for scene_key, pg_name in [
            ("xyz", "xyz"),
            ("scales", "scales"),
            ("rotations", "rotations"),
            ("opacity", "opacity"),
            ("shs", "shs"),
        ]:
            raw = scene_data[scene_key].detach().clone()
            p = nn.Parameter(raw)

            quats = F.normalize(p if pg_name == "rotations" else scene_data["rotations"].detach(), dim=-1)
            scales = torch.exp(p if pg_name == "scales" else scene_data["scales"].detach()).contiguous()
            ops = (torch.sigmoid(p).squeeze(-1) if pg_name == "opacity" else
                   torch.sigmoid(scene_data["opacity"].detach()).squeeze(-1))
            colors = p if pg_name == "shs" else scene_data["shs"].detach()
            means = p if pg_name == "xyz" else scene_data["xyz"].detach()

            rendered, _, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=ops, colors=colors,
                viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                width=w, height=h, tile_size=16, sh_degree=3,
                packed=True, radius_clip=rc, render_mode="RGB",
            )
            out = rendered[0].clamp(0, 1)
            loss = F.mse_loss(out, gt_image)
            loss.backward()

            g = p.grad
            info = {
                "grad_norm": round(float(g.norm()), 12),
                "all_finite": bool(torch.isfinite(g).all()),
                "min": round(float(g.min().item()), 12),
                "max": round(float(g.max().item()), 12),
            }
            print(f"    {pg_name}: norm={info['grad_norm']:.6e}, finite={info['all_finite']}")
            rc_results[pg_name] = info

            del p
            gc.collect()
            torch.cuda.empty_cache()

        results[f"rclip_{rc}"] = rc_results

    # Compare against rclip=0 (baseline)
    print(f"\n  --- Diff vs rclip=0 (baseline) ---")
    for rc in radius_clip_values[1:]:
        key = f"rclip_{rc}"
        pg = "xyz"
        n_base = results["rclip_0.0"][pg]["grad_norm"]
        n_test = results[key][pg]["grad_norm"]
        rel_diff = abs(n_base - n_test) / max(n_base, n_test) if max(n_base, n_test) > 0 else 0
        print(f"    {key}: xyz_norm={n_test:.6e}, diff_vs_0={rel_diff:.2e}")

    return results


def test_eps2d(scene_data, device="cuda"):
    """Test gradient correctness across different eps2d values."""
    viewmat, K = make_default_camera(device=device)
    w, h = 1920, 1080
    gt_image = torch.rand(h, w, 3, device=device)

    eps2d_values = [0.01, 0.1, 0.3, 0.5]

    results = {}
    for eps2d in eps2d_values:
        print(f"\n  === eps2d={eps2d} ===")
        eps_results = {}

        for scene_key, pg_name in [
            ("xyz", "xyz"),
            ("scales", "scales"),
            ("rotations", "rotations"),
            ("opacity", "opacity"),
            ("shs", "shs"),
        ]:
            raw = scene_data[scene_key].detach().clone()
            p = nn.Parameter(raw)

            quats = F.normalize(p if pg_name == "rotations" else scene_data["rotations"].detach(), dim=-1)
            scales = torch.exp(p if pg_name == "scales" else scene_data["scales"].detach()).contiguous()
            ops = (torch.sigmoid(p).squeeze(-1) if pg_name == "opacity" else
                   torch.sigmoid(scene_data["opacity"].detach()).squeeze(-1))
            colors = p if pg_name == "shs" else scene_data["shs"].detach()
            means = p if pg_name == "xyz" else scene_data["xyz"].detach()

            rendered, _, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=ops, colors=colors,
                viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                width=w, height=h, tile_size=16, sh_degree=3,
                packed=True, eps2d=eps2d, render_mode="RGB",
            )
            out = rendered[0].clamp(0, 1)
            loss = F.mse_loss(out, gt_image)
            loss.backward()

            g = p.grad
            info = {
                "grad_norm": round(float(g.norm()), 12),
                "all_finite": bool(torch.isfinite(g).all()),
            }
            print(f"    {pg_name}: norm={info['grad_norm']:.6e}, finite={info['all_finite']}")
            eps_results[pg_name] = info

            del p
            gc.collect()
            torch.cuda.empty_cache()

        results[f"eps2d_{eps2d}"] = eps_results

    # Compare against eps2d=0.1 (default)
    print(f"\n  --- Diff vs eps2d=0.1 (default) ---")
    for eps2d in eps2d_values:
        if eps2d == 0.1:
            continue
        key = f"eps2d_{eps2d}"
        pg = "xyz"
        n_base = results["eps2d_0.1"][pg]["grad_norm"]
        n_test = results[key][pg]["grad_norm"]
        rel_diff = abs(n_base - n_test) / max(n_base, n_test) if max(n_base, n_test) > 0 else 0
        print(f"    {key}: xyz_norm={n_test:.6e}, diff_vs_0.1={rel_diff:.2e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="EPIC-06 P1: Gradient check for M4/M5")
    parser.add_argument("--module", nargs="+", choices=["M4", "M5"], default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gaussians", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modules = args.module
    if args.all:
        modules = ["M4", "M5"]

    if not modules:
        print("Specify --module M4, --module M5, or --all")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nGenerating scene: {args.gaussians} Gaussians")
    scene_data = generate_scene(args.gaussians, device, args.seed)

    output_dir = REPO_ROOT / "results" / "epic05" / "gradient"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if "M4" in modules:
        print(f"\n{'=' * 60}")
        print(f"  M4: radius_clip gradient comparison")
        print(f"{'=' * 60}")
        r4 = test_radius_clip(scene_data, device)
        path = output_dir / f"gradcheck_M4_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(r4, f, indent=2)
        print(f"\n  Saved: {path}")

    if "M5" in modules:
        print(f"\n{'=' * 60}")
        print(f"  M5: eps2d gradient comparison")
        print(f"{'=' * 60}")
        r5 = test_eps2d(scene_data, device)
        path = output_dir / f"gradcheck_M5_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(r5, f, indent=2)
        print(f"\n  Saved: {path}")

    print(f"\nDone! Results in {output_dir}")


if __name__ == "__main__":
    main()
