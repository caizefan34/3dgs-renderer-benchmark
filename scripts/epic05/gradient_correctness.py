#!/usr/bin/env python3
"""
EPIC-06 P0: Gradient Correctness for Tile Size (gradcheck + finite difference).

This is the SINGLE HIGHEST PRIORITY experiment after the Phase 6 research
alignment audit. It verifies that tile_size=16 and tile_size=32 produce
numerically correct gradients for all parameter groups.

Protocol:
    1. torch.autograd.gradcheck on gsplat.rasterization() at tile16 and tile32
    2. Finite-difference (central, eps=1e-4, 1e-5) for:
       - means (positions)
       - scales
       - rotations (quaternions)
       - opacity
       - shs (SH coefficients)
    3. Synthetic scene (50K Gaussians), single camera, 1080p
    4. Test both packed and dense modes
    5. Report: max_abs_error, max_relative_error, mean_relative_error

Usage:
    python scripts/epic05/gradient_correctness.py
    python scripts/epic05/gradient_correctness.py --dense --quick
    python scripts/epic05/gradient_correctness.py --scene room --tile-sizes 16 32

Output:
    results/epic05/gradient/gradcheck_<config>_<timestamp>.json
"""

import argparse
import gc
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
import torch.utils.cpp_extension as cpp_ext
cpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')

_msvc_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
for _p in [_msvc_dir, _cuda_bin]:
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
os.environ["CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
from gsplat import rasterization


# ===================================================================
# Rasterization wrapper with parameter-group isolation
# ===================================================================

class RasterizationWrapper(nn.Module):
    """Wraps gsplat.rasterization() for per-parameter gradient checking.

    Holds camera and configuration fixed; exposes per-parameter-group
    forward passes so gradients can be checked individually.
    """

    def __init__(
        self,
        tile_size: int = 16,
        packed: bool = True,
        sh_degree: int = 3,
        eps2d: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.tile_size = tile_size
        self.packed = packed
        self.sh_degree = sh_degree
        self.eps2d = eps2d
        self.device = device
        self._camera = None

    def set_camera(self, camera) -> None:
        self._camera = camera

    def forward(self, means, quats, scales, opacities, colors):
        """Direct forward pass with explicit parameters."""
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=self._camera.viewmatrix.unsqueeze(0),
            Ks=self._camera.K.unsqueeze(0),
            width=self._camera.image_width,
            height=self._camera.image_height,
            tile_size=self.tile_size,
            sh_degree=self.sh_degree,
            packed=self.packed,
            eps2d=self.eps2d,
            render_mode="RGB",
        )
        return rendered[0].clamp(0, 1)


# ===================================================================
# Finite-difference gradient comparison
# ===================================================================

def compute_finite_difference(
    wrapper: RasterizationWrapper,
    param_name: str,
    param_tensor: torch.Tensor,
    gt_image: torch.Tensor,
    base_params: dict,
    eps_values=(1e-4, 1e-5),
    n_samples: int = 200,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Compare analytical vs finite-difference gradients for one parameter group.

    Samples `n_samples` parameter indices and computes central-difference
    gradient for each, then compares with the stored analytical gradient.

    Args:
        wrapper: RasterizationWrapper instance (camera already set)
        param_name: One of 'means', 'quats', 'scales', 'opacities', 'colors'
        param_tensor: The nn.Parameter that already has .grad populated
        gt_image: Ground truth image [H, W, 3]
        base_params: Dict of all other params (as plain tensors, not nn.Parameter)
        eps_values: Epsilons for central difference
        n_samples: Number of random indices to test (for speed)
        seed: Random seed for sampling

    Returns:
        dict with FD comparison metrics per epsilon
    """
    torch.manual_seed(seed)

    if param_tensor.grad is None:
        return {"error": "analytical gradient is None — backward not run"}

    flat_ana = param_tensor.grad.detach().flatten()
    total_elems = flat_ana.numel()
    sample_size = min(n_samples, total_elems)
    sample_idx = torch.randperm(total_elems, device=device)[:sample_size]

    results = {}
    for eps in eps_values:
        fd_grads = torch.empty(sample_size, device=device, dtype=torch.float64)
        ana_slice = flat_ana[sample_idx].to(torch.float64)

        for i, idx in enumerate(sample_idx):
            multi_idx = np.unravel_index(idx.cpu().item(), param_tensor.shape)

            # f(x + eps)
            p_plus = param_tensor.detach().clone()
            p_plus[multi_idx] += eps
            # Squeeze opacity [N,1] -> [N] for rasterization API
            p_plus_input = p_plus.squeeze(-1) if param_name == "opacities" else p_plus
            with torch.no_grad():
                params_plus = dict(base_params)
                params_plus[param_name] = p_plus_input
                out_plus = wrapper(**params_plus)
                loss_plus = F.mse_loss(out_plus, gt_image)

            # f(x - eps)
            p_minus = param_tensor.detach().clone()
            p_minus[multi_idx] -= eps
            p_minus_input = p_minus.squeeze(-1) if param_name == "opacities" else p_minus
            with torch.no_grad():
                params_minus = dict(base_params)
                params_minus[param_name] = p_minus_input
                out_minus = wrapper(**params_minus)
                loss_minus = F.mse_loss(out_minus, gt_image)

            fd_grads[i] = (loss_plus - loss_minus) / (2 * eps)

        abs_errors = (fd_grads - ana_slice).abs()
        rel_errors = abs_errors / (ana_slice.abs() + 1e-12)

        results[f"eps_{eps:.0e}"] = {
            "eps": eps,
            "n_samples": sample_size,
            "max_abs_error": float(abs_errors.max()),
            "mean_abs_error": float(abs_errors.mean()),
            "median_abs_error": float(abs_errors.median()),
            "max_relative_error": float(rel_errors.max()),
            "mean_relative_error": float(rel_errors.mean()),
            "median_relative_error": float(rel_errors.median()),
            "ana_grad_norm": float(flat_ana.norm()),
            "fd_grad_norm": float(fd_grads.norm()),
        }

    return results


# ===================================================================
# Autograd gradcheck
# ===================================================================

def run_gradcheck(
    wrapper: RasterizationWrapper,
    param_name: str,
    scene_data: dict,
    camera,
    gt_image: torch.Tensor,
    param_group: str,
    device: str = "cuda",
) -> dict:
    """Run torch.autograd.gradcheck on a cropped region.

    Uses a 64x64 center crop for tractability, and checks only the first
    20 elements (full-parameter gradcheck is O(n^2)).
    """
    crop_size = 64
    h, w = gt_image.shape[:2]
    h_start, w_start = (h - crop_size) // 2, (w - crop_size) // 2
    gt_cropped = gt_image[h_start:h_start+crop_size, w_start:w_start+crop_size]

    # Build the closure
    def make_func(gt_crop):
        def func(*args):
            means = args[0] if param_name in ("means", "xyz") else scene_data["xyz"].detach()
            quats_raw = args[0] if param_name in ("quats", "rotations") else scene_data["rotations"].detach()
            scales_raw = args[0] if param_name in ("scales",) else scene_data["scales"].detach()
            ops_raw = args[0] if param_name in ("opacity",) else scene_data["opacity"].detach()
            colors_raw = args[0] if param_name in ("shs", "colors") else scene_data["shs"].detach()

            # Normalize/activate as needed
            quats = F.normalize(quats_raw, dim=-1) if param_name in ("quats", "rotations") else \
                    F.normalize(scene_data["rotations"].detach(), dim=-1)
            scales = torch.exp(scales_raw) if param_name in ("scales",) else \
                     torch.exp(scene_data["scales"].detach())
            ops = torch.sigmoid(ops_raw).squeeze(-1) if param_name in ("opacity",) else \
                  torch.sigmoid(scene_data["opacity"].detach()).squeeze(-1)
            colors = colors_raw

            out = wrapper(means, quats, scales, ops, colors)
            out_crop = out[h_start:h_start+crop_size, w_start:w_start+crop_size]
            return F.mse_loss(out_crop, gt_crop)

        return func

    func = make_func(gt_cropped)

    # Map scene_key to correct tensor for extraction
    param_tensor_map = {
        "xyz": scene_data["xyz"],
        "means": scene_data["xyz"],
        "rotations": scene_data["rotations"],
        "quats": scene_data["rotations"],
        "scales": scene_data["scales"],
        "opacity": scene_data["opacity"],
        "shs": scene_data["shs"],
        "colors": scene_data["shs"],
    }
    if param_group not in param_tensor_map:
        return {"error": f"unknown param_group: {param_group}"}
    param_raw = param_tensor_map[param_group].detach().clone()[:20].requires_grad_(True)

    result = {"param_group": param_group, "crop_size": crop_size, "n_checked": 20}

    try:
        gradcheck_ok = torch.autograd.gradcheck(
            func, (param_raw,),
            eps=1e-4, atol=1e-3, rtol=1e-3,
            raise_exception=False, check_sparse_nans=True, nondet_tol=1e-5,
        )
        result["gradcheck_passed"] = bool(gradcheck_ok)
    except Exception as e:
        result["gradcheck_passed"] = False
        result["gradcheck_error"] = str(e)

    try:
        gradgradcheck_ok = torch.autograd.gradgradcheck(
            func, (param_raw,),
            eps=1e-4, atol=1e-3, rtol=1e-3,
            raise_exception=False, nondet_tol=1e-5,
        )
        result["gradgradcheck_passed"] = bool(gradgradcheck_ok)
    except Exception as e:
        result["gradgradcheck_passed"] = False
        result["gradgradcheck_error"] = str(e)

    return result


# ===================================================================
# Utility: build base params from scene_data
# ===================================================================

def build_base_params(scene_data: dict, exclude: str, device: str = "cuda") -> dict:
    """Build the 'other params' dict for the wrapper.forward() call.

    Args:
        scene_data: Scene dict with raw tensors
        exclude: Which parameter to exclude (the tested one, e.g. 'xyz', 'shs')

    Returns:
        dict with keys: means, quats, scales, opacities, colors
    """
    quats = F.normalize(scene_data["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene_data["scales"]).contiguous()
    ops = torch.sigmoid(scene_data["opacity"]).squeeze(-1).contiguous()  # [N] for rasterization
    colors = scene_data["shs"].contiguous()

    params = {
        "means": scene_data["xyz"].contiguous().detach(),
        "quats": quats.detach(),
        "scales": scales.detach(),
        "opacities": ops.detach(),
        "colors": colors.detach(),
    }

    # Remove the excluded one (it will be provided separately for grad)
    exclude_map = {"xyz": "means", "rotations": "quats", "scales": "scales",
                   "opacity": "opacities", "shs": "colors"}
    excl_key = exclude_map.get(exclude, exclude)
    if excl_key in params:
        del params[excl_key]

    return params


# ===================================================================
# Scene / camera helpers
# ===================================================================

def generate_synthetic_scene(num_points: int = 50000, device: str = "cuda", seed: int = 42):
    """Generate a synthetic Gaussian scene."""
    torch.manual_seed(seed)
    scene = {
        "xyz": torch.randn(num_points, 3, device=device) * 2.0,
        "rotations": F.normalize(torch.randn(num_points, 4, device=device), dim=-1),
        "scales": torch.log(torch.rand(num_points, 3, device=device) * 0.1 + 0.01),
        "opacity": torch.logit(torch.rand(num_points, 1, device=device) * 0.5 + 0.25),
        "shs": torch.randn(num_points, 16, 3, device=device) * 0.1,
        "num_points": num_points,
    }
    return scene


def load_default_camera(device: str = "cuda"):
    """Create a default camera."""
    from benchmark_framework.cameras import Camera
    import numpy as np
    import math

    w, h = 1920, 1080
    fx, fy = w * 0.6, h * 0.6
    tan_fov_x = w / (2.0 * fx)
    tan_fov_y = h / (2.0 * fy)
    near, far = 0.01, 100.0

    # Camera at (0, 0, 5) looking at origin
    cam_to_world = np.eye(4, dtype=np.float32)
    cam_to_world[2, 3] = 5.0  # position at z=5
    viewmatrix = torch.tensor(
        np.linalg.inv(cam_to_world).astype(np.float32),
        dtype=torch.float32, device=device,
    )
    cam_pos = torch.tensor([0, 0, 5], dtype=torch.float32, device=device)

    projmatrix = torch.tensor([
        [1.0 / tan_fov_x, 0, 0, 0],
        [0, 1.0 / tan_fov_y, 0, 0],
        [0, 0, far / (far - near), -far * near / (far - near)],
        [0, 0, 1, 0],
    ], dtype=torch.float32, device=device)

    K = torch.tensor([
        [fx, 0, w / 2.0],
        [0, fy, h / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)

    camera = Camera(
        image_width=w,
        image_height=h,
        fov_x=2.0 * math.atan(tan_fov_x),
        fov_y=2.0 * math.atan(tan_fov_y),
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        camera_center=cam_pos,
        world_view_transform=viewmatrix.T.contiguous(),
        full_proj_transform=(projmatrix @ viewmatrix).T.contiguous(),
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        K=K,
    )
    return camera


# ===================================================================
# Single-configuration test
# ===================================================================

def test_one_config(
    scene_data: dict,
    camera,
    tile_size: int = 16,
    packed: bool = True,
    sh_degree: int = 3,
    seed: int = 42,
    device: str = "cuda",
    quick: bool = False,
    skip_gradcheck: bool = False,
) -> dict:
    """Run complete gradient correctness for one (tile_size, packed) config."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    wrapper = RasterizationWrapper(tile_size=tile_size, packed=packed,
                                   sh_degree=sh_degree, device=device)
    wrapper.set_camera(camera)

    gt_image = torch.rand((camera.image_height, camera.image_width, 3), device=device)

    result = {
        "tile_size": tile_size,
        "packed": packed,
        "sh_degree": sh_degree,
        "seed": seed,
        "camera": {"width": camera.image_width, "height": camera.image_height},
        "scene": {"num_gaussians": scene_data["num_points"]},
        "param_groups": {},
    }

    # Map: (scene_key, pg_display_name, wrapper_param_name)
    param_map = [
        ("xyz", "xyz", "means"),
        ("scales", "scales", "scales"),         # log-scale, we pass raw and exp inside wrapper
        ("rotations", "rotations", "quats"),    # quaternions
        ("opacity", "opacity", "opacities"),    # logit-scale -> wrapper expects 'opacities'
        ("shs", "shs", "colors"),               # SH coefficients -> colors in gsplat API
    ]

    if quick:
        param_map = [("xyz", "xyz", "means"), ("shs", "shs", "colors")]

    for scene_key, pg_name, wrapper_name in param_map:
        print(f"\n  Parameter group: {pg_name} (tile_size={tile_size}, packed={packed})")

        # Create the test parameter (requires_grad)
        raw_tensor = scene_data[scene_key].detach().clone()
        test_param = nn.Parameter(raw_tensor)

        # Map to the inputs rasterization() expects
        quats = F.normalize(test_param if pg_name == "rotations" else scene_data["rotations"].detach(), dim=-1)
        scales = torch.exp(test_param if pg_name == "scales" else scene_data["scales"].detach()).contiguous()
        ops = (torch.sigmoid(test_param).squeeze(-1) if pg_name == "opacity" else
               torch.sigmoid(scene_data["opacity"].detach()).squeeze(-1))  # [N] for rasterization
        colors = (test_param if pg_name == "shs" else scene_data["shs"].detach())
        means = test_param if pg_name == "xyz" else scene_data["xyz"].detach()

        # Ensure contiguous
        quats = quats.contiguous()
        scales = scales.contiguous()
        ops = ops.contiguous()
        colors = colors.contiguous()
        means = means.contiguous()

        # Forward + backward
        output = wrapper(means, quats, scales, ops, colors)
        loss = F.mse_loss(output, gt_image)
        loss.backward()
        torch.cuda.synchronize()

        pg_info = {}

        if test_param.grad is None:
            print(f"    !!  No gradient!")
            pg_info["has_gradient"] = False
            pg_info["error"] = "no_gradient"
            result["param_groups"][pg_name] = pg_info
            del test_param
            gc.collect()
            torch.cuda.empty_cache()
            continue

        pg_info["has_gradient"] = True
        g = test_param.grad
        pg_info["analytical_grad"] = {
            "norm": round(float(g.norm()), 12),
            "min": round(float(g.min()), 12),
            "max": round(float(g.max()), 12),
            "mean": round(float(g.mean()), 12),
            "std": round(float(g.std()), 12),
            "num_elements": g.numel(),
            "shape": list(g.shape),
            "all_finite": bool(torch.isfinite(g).all()),
        }
        print(f"    Grad: norm={float(g.norm()):.6e}, range=[{float(g.min()):.6e}, {float(g.max()):.6e}], "
              f"finite={torch.isfinite(g).all().item()}")

        # ---- Finite-difference ----
        print(f"    Finite-difference comparison (n=100)...")
        base = build_base_params(scene_data, pg_name, device)
        # Re-add the test param under its wrapper name
        if pg_name == "xyz":
            base["means"] = means.detach()
        elif pg_name == "rotations":
            base["quats"] = quats.detach()
        elif pg_name == "scales":
            base["scales"] = scales.detach()
        elif pg_name == "opacity":
            base["opacities"] = ops.detach()
        elif pg_name == "shs":
            base["colors"] = colors.detach()

        try:
            fd_result = compute_finite_difference(
                wrapper=wrapper,
                param_name=wrapper_name,
                param_tensor=test_param,
                gt_image=gt_image,
                base_params=base,
                eps_values=(1e-4, 1e-5),
                n_samples=100,
                seed=seed,
                device=device,
            )
            pg_info["finite_difference"] = fd_result
            for ek, ev in fd_result.items():
                if isinstance(ev, dict) and "max_abs_error" in ev:
                    print(f"      {ek}: max_abs={ev['max_abs_error']:.6e}, "
                          f"max_rel={ev['max_relative_error']:.6e}")
        except Exception as e:
            print(f"    !!  FD failed: {e}")
            pg_info["finite_difference"] = {"error": str(e)}

        # ---- Gradcheck ----
        if not skip_gradcheck and test_param.numel() >= 20:
            print(f"    Gradcheck (first 20 elements, 64x64 crop)...")
            try:
                gc_result = run_gradcheck(
                    wrapper, wrapper_name, scene_data, camera, gt_image,
                    pg_name, device,
                )
                pg_info["gradcheck"] = gc_result
                print(f"      gradcheck passed: {gc_result.get('gradcheck_passed')}")
            except Exception as e:
                pg_info["gradcheck"] = {"error": str(e)}
        else:
            pg_info["gradcheck"] = {"note": "skipped"}

        result["param_groups"][pg_name] = pg_info

        # Cleanup
        del test_param
        gc.collect()
        torch.cuda.empty_cache()

    return result


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EPIC-06 P0: Gradient Correctness for Tile Size"
    )
    parser.add_argument("--gaussians", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--dense", action="store_true",
                        help="Also test packed=False (dense) mode")
    parser.add_argument("--quick", action="store_true",
                        help="Only test means and shs (fastest groups)")
    parser.add_argument("--no-gradcheck", action="store_true",
                        help="Skip gradcheck (slow)")
    args = parser.parse_args()

    packed_modes = [True]
    if args.dense:
        packed_modes.append(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    # Scene
    print(f"\nGenerating synthetic scene: {args.gaussians} Gaussians")
    scene_data = generate_synthetic_scene(args.gaussians, device, args.seed)
    camera = load_default_camera(device)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "results" / "epic05" / "gradient"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for tile_size in args.tile_sizes:
        for packed in packed_modes:
            mode = "packed" if packed else "dense"
            key = f"tile{tile_size}_{mode}"
            print(f"\n{'='*70}")
            print(f"  Config: tile_size={tile_size}, mode={mode}")
            print(f"{'='*70}")

            r = test_one_config(
                scene_data, camera,
                tile_size=tile_size, packed=packed,
                seed=args.seed, device=device,
                quick=args.quick, skip_gradcheck=args.no_gradcheck,
            )
            all_results[key] = r

            ipath = output_dir / f"gradcheck_{key}_{timestamp}.json"
            with open(ipath, "w") as f:
                json.dump(r, f, indent=2, cls=_Encoder)
            print(f"  Saved: {ipath}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    header = f"{'Config':<20} {'Param':<12} {'Grad?':<8} {'Gradcheck':<12} {'MaxRel(e-4)':<14} {'MaxAbs(e-4)':<14}"
    print(header)
    print("-" * 80)

    for key, r in all_results.items():
        for pg, pgd in r["param_groups"].items():
            if not pgd.get("has_gradient", False):
                print(f"{key:<20} {pg:<12} {'NO':<8} {'-':<12} {'-':<14} {'-':<14}")
                continue
            gc_p = pgd.get("gradcheck", {}).get("gradcheck_passed")
            gc_s = "PASS" if gc_p else ("FAIL" if gc_p is False else "N/A")
            fd = pgd.get("finite_difference", {})
            me = fd.get("eps_1e-04", {}).get("max_relative_error", None)
            ma = fd.get("eps_1e-04", {}).get("max_abs_error", None)
            rel_s = f"{me:.4e}" if me else "-"
            abs_s = f"{ma:.6e}" if ma else "-"
            print(f"{key:<20} {pg:<12} {'YES':<8} {gc_s:<12} {rel_s:<14} {abs_s:<14}")

    # Overall verdict
    print(f"\n{'='*70}")
    any_fail = False
    for key, r in all_results.items():
        for pg, pgd in r["param_groups"].items():
            fd = pgd.get("finite_difference", {}).get("eps_1e-04", {})
            if fd.get("max_relative_error", 0) > 1e-3:
                any_fail = True
                print(f"  !!  {key}/{pg}: max_relative_error={fd['max_relative_error']:.6e} > 1e-3")
    if not any_fail:
        print(f"  [OK] All tested parameter groups within tolerance (max_rel < 1e-3)")

    summary = {
        "experiment_id": "epic06-gradient-correctness-v1",
        "date": date.today().isoformat(),
        "timestamp": timestamp,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": {
            "num_gaussians": args.gaussians, "seed": args.seed,
            "tile_sizes": args.tile_sizes, "packed_modes": packed_modes,
            "quick": args.quick, "gradcheck": not args.no_gradcheck,
        },
        "results": all_results,
    }
    spath = output_dir / f"gradcheck_summary_{timestamp}.json"
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2, cls=_Encoder)
    print(f"\nFull results: {spath}")


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, (torch.Tensor,)): return o.cpu().tolist()
        if isinstance(o, (bytes,)): return o.decode()
        return super().default(o)


if __name__ == "__main__":
    main()
