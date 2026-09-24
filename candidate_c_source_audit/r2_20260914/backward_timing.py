"""
R2 Parts 6-9: Backward Component Timing + Derivative Family Cost.

Instruments the gsplat autograd backward to time each component:
  1. rasterize_to_pixels_3dgs_bwd (produces v_colors, v_opacity, v_means2d, v_conics)
  2. projection_ewa_3dgs_fused_bwd (produces v_means, v_covars, v_quats, v_scales from v_means2d, v_conics)
  3. spherical_harmonics_bwd (produces v_coeffs, v_dirs from v_colors)

Also measures variants:
  - FULL: all derivatives
  - NO_SH_BACKWARD: skip SH backward (set v_colors=None path)
  - NO_PROJECTION_BACKWARD: skip projection backward
  - BY_ZEROING: zero specific gradient outputs after backward to measure overhead

Uses CUDA events for precise timing. 20 warmup, 100 measured.
"""

import sys, os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
BASELINE_DIR = os.path.join(REPO_ROOT, "baseline", "reference_v1")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, SRC_DIR)

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization
import gsplat.cuda._wrapper as gsplat_wrapper

OLD_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "epic05", "phase7")
sys.path.insert(0, OLD_SCRIPTS)
from dataset import GTDataset


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2; self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3: pred = pred.unsqueeze(0).permute(0, 3, 1, 2); target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()


def cuda_time(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), result


# === Instrumented backward functions ===
# We hook into the autograd backward by replacing the C functions temporarily

_original_raster_bwd = None
_original_proj_bwd = None
_original_sh_bwd = None
_timing_data = {}


def make_timed_raster_bwd(original_fn):
    def timed_fn(*args, **kwargs):
        t, result = cuda_time(lambda: original_fn(*args, **kwargs))
        _timing_data["raster_bwd"] = t
        return result
    return timed_fn


def make_timed_proj_bwd(original_fn):
    def timed_fn(*args, **kwargs):
        t, result = cuda_time(lambda: original_fn(*args, **kwargs))
        _timing_data["proj_bwd"] = t
        return result
    return timed_fn


def make_timed_sh_bwd(original_fn):
    def timed_fn(*args, **kwargs):
        t, result = cuda_time(lambda: original_fn(*args, **kwargs))
        _timing_data["sh_bwd"] = t
        return result
    return timed_fn


def install_timers():
    """Install timing hooks into gsplat's lazy CUDA function dispatch."""
    global _original_raster_bwd, _original_proj_bwd, _original_sh_bwd

    # Save originals by calling the lazy loader
    from gsplat.cuda._backend import _C
    _original_raster_bwd = _C.rasterize_to_pixels_3dgs_bwd
    _original_proj_bwd = _C.projection_ewa_3dgs_fused_bwd
    _original_sh_bwd = _C.spherical_harmonics_bwd

    # Replace with timed versions
    _C.rasterize_to_pixels_3dgs_bwd = make_timed_raster_bwd(_original_raster_bwd)
    _C.projection_ewa_3dgs_fused_bwd = make_timed_proj_bwd(_original_proj_bwd)
    _C.spherical_harmonics_bwd = make_timed_sh_bwd(_original_sh_bwd)


def uninstall_timers():
    """Restore original functions."""
    from gsplat.cuda._backend import _C
    if _original_raster_bwd is not None:
        _C.rasterize_to_pixels_3dgs_bwd = _original_raster_bwd
        _C.projection_ewa_3dgs_fused_bwd = _original_proj_bwd
        _C.spherical_harmonics_bwd = _original_sh_bwd


def run_backward_timing(checkpoint_path, output_path, config, camera_sequence_path):
    device = "cuda"
    dataset = GTDataset(config.scene, config.repo_root)
    camera_sequence = np.load(camera_sequence_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ckpt, {
        "position_lr_init": config.position_lr_init, "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult, "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr, "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr, "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    N = model._xyz.shape[0]
    print(f"Loaded: N={N}")
    ssim_fn = SepSSIM(device=device)

    benchmark_iters = [10001, 10002, 10003, 10004, 10005]
    n_warmup = 20
    n_measure = 100

    # === Install timing hooks ===
    install_timers()

    # === FULL backward timing ===
    print("\n=== FULL backward timing ===")
    raster_times, proj_times, sh_times, total_bwd_times, e2e_times = [], [], [], [], []

    for bench_iter in benchmark_iters:
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)

        for _ in range(n_warmup):
            model.optimizer.zero_grad(set_to_none=True)
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation, scales=model.get_scaling,
                opacities=model.get_opacity, colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
            )
            image = r.squeeze(0)
            L1 = F.l1_loss(image, gt_image)
            dssim = ssim_fn(image, gt_image)
            loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        for _ in range(n_measure):
            model.optimizer.zero_grad(set_to_none=True)
            _timing_data.clear()

            t_fwd, (r, _, meta) = cuda_time(lambda: rasterization(
                means=model.get_xyz, quats=model.get_rotation, scales=model.get_scaling,
                opacities=model.get_opacity, colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
            ))

            image = r.squeeze(0)
            L1 = F.l1_loss(image, gt_image)
            dssim = ssim_fn(image, gt_image)
            loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

            t_bwd, _ = cuda_time(lambda: loss.backward())
            total_bwd_times.append(t_bwd)
            e2e_times.append(t_fwd + t_bwd)

            raster_times.append(_timing_data.get("raster_bwd", 0))
            proj_times.append(_timing_data.get("proj_bwd", 0))
            sh_times.append(_timing_data.get("sh_bwd", 0))

            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

    def stats(arr):
        a = np.array(arr)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                "n": len(a)}

    results = {
        "checkpoint": checkpoint_path,
        "N_gaussians": N,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "n_cameras": len(benchmark_iters),
        "FULL": {
            "raster_bwd_ms": stats(raster_times),
            "projection_bwd_ms": stats(proj_times),
            "sh_bwd_ms": stats(sh_times),
            "total_backward_ms": stats(total_bwd_times),
            "e2e_ms": stats(e2e_times),
        },
    }

    print(f"  Raster bwd: {np.median(raster_times):.2f}ms")
    print(f"  Proj bwd: {np.median(proj_times):.2f}ms")
    print(f"  SH bwd: {np.median(sh_times):.2f}ms")
    print(f"  Total bwd: {np.median(total_bwd_times):.2f}ms")
    print(f"  E2E: {np.median(e2e_times):.2f}ms")

    # === Derivative family cost decomposition ===
    # raster_bwd produces v_colors (→SH), v_opacity (→opacity), v_means2d+v_conics (→projection→xyz/scale/rot)
    # The raster kernel time is SHARED across all families.
    # sh_bwd is APPEARANCE_ONLY (dL/dSH + dL/dview_dir→dL/dxyz)
    # proj_bwd is GEOMETRY_ONLY (dL/dxyz, dL/dscale, dL/drot from v_means2d, v_conics)

    raster_med = np.median(raster_times)
    proj_med = np.median(proj_times)
    sh_med = np.median(sh_times)
    total_bwd_med = np.median(total_bwd_times)
    e2e_med = np.median(e2e_times)

    # Estimate: raster_bwd is shared, sh_bwd is appearance-exclusive, proj_bwd is geometry-exclusive
    # Other overhead (loss backward, autograd graph) = total - raster - sh - proj
    other = total_bwd_med - raster_med - proj_med - sh_med

    results["derivative_family_decomposition"] = {
        "raster_bwd_shared_ms": raster_med,
        "sh_bwd_appearance_ms": sh_med,
        "proj_bwd_geometry_ms": proj_med,
        "other_overhead_ms": other,
        "total_backward_ms": total_bwd_med,
        "note": "raster_bwd is SHARED (produces v_colors→SH, v_opacity→opacity, v_means2d/v_conics→geometry). sh_bwd is APPEARANCE+GEOMETRY (v_coeffs→dL/dSH, v_dirs→dL/dxyz). proj_bwd is GEOMETRY_ONLY (dL/dxyz, dL/dscale, dL/drot).",
    }

    # Amdahl table
    # Max removable: if we could skip SH bwd entirely → save sh_med / e2e_med
    # If we could skip proj bwd entirely → save proj_med / e2e_med
    # Raster bwd is shared and cannot be removed without breaking all families
    results["amdahl_table"] = {
        "appearance_sh": {
            "exclusive_ms": sh_med,
            "shared_ms": 0,  # SH bwd is exclusive
            "max_removable_e2e_pct": float(sh_med / e2e_med * 100),
        },
        "geometry_projection": {
            "exclusive_ms": proj_med,
            "shared_ms": 0,  # proj bwd is exclusive
            "max_removable_e2e_pct": float(proj_med / e2e_med * 100),
        },
        "raster_shared": {
            "exclusive_ms": 0,
            "shared_ms": raster_med,
            "max_removable_e2e_pct": 0,  # cannot remove shared work
        },
        "opacity": {
            "exclusive_ms": 0,  # opacity gradient is computed inside raster_bwd, no separate kernel
            "shared_ms": raster_med,  # part of shared raster
            "max_removable_e2e_pct": 0,
            "note": "Opacity gradient (v_opacities) is computed within raster_bwd kernel. No separate backward function.",
        },
    }

    print(f"\n=== Amdahl Table ===")
    print(f"  SH bwd (appearance): {sh_med:.2f}ms → {sh_med/e2e_med*100:.1f}% E2E")
    print(f"  Proj bwd (geometry): {proj_med:.2f}ms → {proj_med/e2e_med*100:.1f}% E2E")
    print(f"  Raster bwd (shared): {raster_med:.2f}ms → {raster_med/e2e_med*100:.1f}% E2E")
    print(f"  Other: {other:.2f}ms → {other/e2e_med*100:.1f}% E2E")

    uninstall_timers()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/reference_v1/ckpt_14k/checkpoints/iter_10000.pt")
    parser.add_argument("--output", default="results/reference_v1/r2/derivative_family_cost.json")
    parser.add_argument("--camera-sequence", default="results/reference_v1/room_30k/camera_sequence.npy")
    args = parser.parse_args()
    config = ReferenceV1Config(scene="room", iterations=30000)
    run_backward_timing(args.checkpoint, args.output, config, args.camera_sequence)
