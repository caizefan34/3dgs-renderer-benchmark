#!/usr/bin/env python3
"""
EPIC-06 P1: Gradient Correctness for M2 (packed/dense) and M3 (SH degree).

Extends the gradient correctness infrastructure to additional optimization modules.

Usage:
    python scripts/epic05/gradient_check_modules.py --module M2
    python scripts/epic05/gradient_check_modules.py --module M3
    python scripts/epic05/gradient_check_modules.py --module M2 --module M3 --all
"""
import argparse
import gc
import json
import os
import sys
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Environment
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


class SimpleRasterWrapper(nn.Module):
    """Minimal wrapper for per-parameter gradient checking."""

    def __init__(self, tile_size=16, packed=True, sh_degree=3, eps2d=0.1, device="cuda"):
        super().__init__()
        self.tile_size = tile_size
        self.packed = packed
        self.sh_degree = sh_degree
        self.eps2d = eps2d
        self.device = device
        self.camera = None

    def set_camera(self, viewmat, K, width, height):
        self.viewmat = viewmat.unsqueeze(0)
        self.K = K.unsqueeze(0)
        self.width = width
        self.height = height

    def forward(self, means, quats, scales, opacities, colors):
        rendered, _, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=self.viewmat, Ks=self.K,
            width=self.width, height=self.height,
            tile_size=self.tile_size,
            sh_degree=self.sh_degree,
            packed=self.packed,
            eps2d=self.eps2d,
            render_mode="RGB",
        )
        return rendered[0].clamp(0, 1)


def make_default_camera(w=1920, h=1080, device="cuda"):
    import math
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


def run_gradcheck_one(scene_data, viewmat, K, w, h, n_gaussians_check=10, crop_size=64, device="cuda"):
    """Run gradcheck for a given wrapper config, checking all 5 param groups."""
    from gsplat import rasterization

    gt_image = torch.rand(h, w, 3, device=device)

    param_configs = [
        ("xyz", scene_data["xyz"][:n_gaussians_check]),
        ("scales", scene_data["scales"][:n_gaussians_check]),
        ("rotations", scene_data["rotations"][:n_gaussians_check]),
        ("opacity", scene_data["opacity"][:n_gaussians_check]),
        ("shs", scene_data["shs"][:n_gaussians_check]),
    ]

    h_start, w_start = (h - crop_size) // 2, (w - crop_size) // 2

    results = {}
    for pg_name, raw_tensor in param_configs:
        p = raw_tensor.detach().clone().requires_grad_(True)

        def closure_factory(pg, p_tensor):
            def closure(*args):
                x = args[0]
                # Build inputs: make only the tested param = x, others detached
                means_in = x if pg == "xyz" else scene_data["xyz"][:n_gaussians_check].detach()
                q_in = F.normalize(x if pg == "rotations" else scene_data["rotations"][:n_gaussians_check].detach(), dim=-1)
                s_in = torch.exp(x if pg == "scales" else scene_data["scales"][:n_gaussians_check].detach())
                o_in = (torch.sigmoid(x) if pg == "opacity" else
                        torch.sigmoid(scene_data["opacity"][:n_gaussians_check].detach())).squeeze(-1)
                c_in = x if pg == "shs" else scene_data["shs"][:n_gaussians_check].detach()

                rendered, _, _ = rasterization(
                    means=means_in, quats=q_in, scales=s_in,
                    opacities=o_in, colors=c_in,
                    viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                    width=w, height=h,
                    tile_size=wrapper.tile_size, sh_degree=wrapper.sh_degree,
                    packed=wrapper.packed, eps2d=wrapper.eps2d,
                    render_mode="RGB",
                )
                out = rendered[0].clamp(0, 1)
                out_crop = out[h_start:h_start+crop_size, w_start:w_start+crop_size]
                gt_crop = gt_image[h_start:h_start+crop_size, w_start:w_start+crop_size]
                return F.mse_loss(out_crop, gt_crop)
            return closure

        closure = closure_factory(pg_name, p)
        try:
            ok = torch.autograd.gradcheck(closure, (p,), eps=1e-4, atol=1e-3, rtol=1e-3,
                                          raise_exception=False, nondet_tol=1e-5)
            results[pg_name] = bool(ok)
        except Exception as e:
            results[pg_name] = f"ERROR: {e}"

    return results


def test_module_M2(scene_data, device="cuda"):
    """Compare gradient correctness between packed and dense modes."""
    viewmat, K = make_default_camera(device=device)
    w, h = 1920, 1080
    gt_image = torch.rand(h, w, 3, device=device)

    results = {}
    for packed in [True, False]:
        mode = "packed" if packed else "dense"
        print(f"\n  === {mode} ===")
        mode_results = {}

        for scene_key, wrapper_name, pg_name in [
            ("xyz", "means", "xyz"),
            ("scales", "scales", "scales"),
            ("rotations", "quats", "rotations"),
            ("opacity", "opacities", "opacity"),
            ("shs", "colors", "shs"),
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
                packed=packed, render_mode="RGB",
            )
            out = rendered[0].clamp(0, 1)
            loss = F.mse_loss(out, gt_image)
            loss.backward()

            g = p.grad
            info = {
                "grad_norm": round(float(g.norm()), 12),
                "all_finite": bool(torch.isfinite(g).all()),
                "shape": list(g.shape),
            }
            print(f"    {pg_name}: norm={info['grad_norm']:.6e}, finite={info['all_finite']}")
            mode_results[pg_name] = info

            del p
            gc.collect()
            torch.cuda.empty_cache()

        results[mode] = mode_results

    # Compare
    print(f"\n  --- Packed vs Dense comparison ---")
    for pg in results["packed"]:
        n1 = results["packed"][pg]["grad_norm"]
        n2 = results["dense"][pg]["grad_norm"]
        max_n = max(n1, n2)
        rel_diff = abs(n1 - n2) / max_n if max_n > 0 else 0.0
        print(f"    {pg}: packed={n1:.6e}, dense={n2:.6e}, rel_diff={rel_diff:.2e}")

    return results


def test_module_M3(scene_data, device="cuda"):
    """Compare gradient correctness between SH degrees: 0, 1, 3."""
    viewmat, K = make_default_camera(device=device)
    w, h = 1920, 1080
    gt_image = torch.rand(h, w, 3, device=device)

    results = {}
    for sh_degree in [0, 1, 3]:
        n_sh = (sh_degree + 1) ** 2
        # Use only the shs columns needed for this degree
        shs_subset = scene_data["shs"][:, :n_sh, :].contiguous()
        print(f"\n  === SH degree={sh_degree} ({n_sh} coeffs) ===")
        deg_results = {}

        for scene_key, pg_name in [("xyz", "xyz"), ("scales", "scales"),
                                    ("rotations", "rotations"), ("opacity", "opacity")]:
            raw = scene_data[scene_key].detach().clone()
            p = nn.Parameter(raw)

            quats = F.normalize(p if pg_name == "rotations" else scene_data["rotations"].detach(), dim=-1)
            scales = torch.exp(p if pg_name == "scales" else scene_data["scales"].detach()).contiguous()
            ops = (torch.sigmoid(p).squeeze(-1) if pg_name == "opacity" else
                   torch.sigmoid(scene_data["opacity"].detach()).squeeze(-1))
            colors = p if pg_name == "shs" else shs_subset
            means = p if pg_name == "xyz" else scene_data["xyz"].detach()

            rendered, _, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=ops, colors=colors,
                viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                width=w, height=h, tile_size=16, sh_degree=sh_degree,
                packed=True, render_mode="RGB",
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
            deg_results[pg_name] = info

            del p
            gc.collect()
            torch.cuda.empty_cache()

        results[f"SH{sh_degree}"] = deg_results

    # Compare across degrees
    print(f"\n  --- SH degree comparison (xyz gradient norm) ---")
    base = results.get("SH3", {}).get("xyz", {}).get("grad_norm", 0)
    for deg in ["SH0", "SH1", "SH3"]:
        n = results.get(deg, {}).get("xyz", {}).get("grad_norm", 0)
        rel_diff = abs(n - base) / max(n, base) if max(n, base) > 0 else 0.0
        print(f"    {deg}: xyz_norm={n:.6e}, diff_vs_SH3={rel_diff:.2e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="EPIC-06 P1: Gradient check for M2/M3")
    parser.add_argument("--module", nargs="+", choices=["M2", "M3"], default=[])
    parser.add_argument("--all", action="store_true", help="Test all modules")
    parser.add_argument("--gaussians", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modules = args.module
    if args.all:
        modules = ["M2", "M3"]

    if not modules:
        print("Specify --module M2, --module M3, or --all")
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

    all_results = {}

    if "M2" in modules:
        print(f"\n{'='*60}")
        print(f"  M2: packed/dense gradient comparison")
        print(f"{'='*60}")
        r2 = test_module_M2(scene_data, device)
        all_results["M2"] = r2
        path = output_dir / f"gradcheck_M2_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(r2, f, indent=2)
        print(f"  Saved: {path}")

    if "M3" in modules:
        print(f"\n{'='*60}")
        print(f"  M3: SH degree gradient comparison")
        print(f"{'='*60}")
        r3 = test_module_M3(scene_data, device)
        all_results["M3"] = r3
        path = output_dir / f"gradcheck_M3_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(r3, f, indent=2)
        print(f"  Saved: {path}")

    print(f"\nDone! Results in {output_dir}")


if __name__ == "__main__":
    main()
