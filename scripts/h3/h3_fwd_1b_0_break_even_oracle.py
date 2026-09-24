#!/usr/bin/env python3
"""H3-FWD-1B-0: Macro Raster Break-even Oracle.

Measures B2 F5 (rasterize_to_pixels) timing, B2 F5 workload statistics,
and existing HiGS macro-tile raster oracle timing.

Does NOT implement a new rasterizer. Performance opportunity oracle only.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import runpy
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

TILE_SIZE = 16
SH_DEGREE = 3
K_SH = (SH_DEGREE + 1) ** 2  # 16

SCENE_CONFIGS = {
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
        "native_w": 3114, "native_h": 2075,
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
        "native_w": 4946, "native_h": 3286,
    },
    "garden": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json",
        "native_w": 5187, "native_h": 3361,
    },
}

# H3-FWD-1A-R2 F4 timing results (from R2 timing JSONs on mx)
R2_F4_TIMING = {
    "room":     {"B2_F4_median_ms": 0.5468159914016724, "Macro_F4_median_ms": 0.8089600205421448},
    "bicycle":  {"B2_F4_median_ms": 0.6696959733963013, "Macro_F4_median_ms": 0.8171520233154297},
    "garden":   {"B2_F4_median_ms": 0.4177919924259186, "Macro_F4_median_ms": 0.45875200629234314},
}

# R2 structural results
R2_STRUCTURAL = {
    "room":     {"N_B2_tile_pairs": 953144, "N_macro_entries": 124017},
    "bicycle":  {"N_B2_tile_pairs": 1412189, "N_macro_entries": 311610},
    "garden":   {"N_B2_tile_pairs": 533928, "N_macro_entries": 66682},
}


def bootstrap(source, core_so):
    """Load the core gsplat module and make it available as gsplat.csrc."""
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core


def load_ply_scene(ply_path, device):
    """Load PLY and return means, quats, scales, opacities, colors."""
    v = PlyData.read(ply_path)["vertex"]
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), device=device, dtype=torch.float32)
    quats = torch.tensor(np.column_stack([v[f"rot_{i}"] for i in range(4)]), device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v[f"scale_{i}"] for i in range(3)]), device=device, dtype=torch.float32))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], device=device, dtype=torch.float32))
    sh = torch.zeros((len(v), K_SH, 3), device=device, dtype=torch.float32)
    sh[:, 0] = torch.tensor(np.column_stack([v[f"f_dc_{i}"] for i in range(3)]), device=device, dtype=torch.float32)
    rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], device=device, dtype=torch.float32) for i in range(45)], 1)
    sh[:, 1:] = rest.reshape(len(v), 3, 15).permute(0, 2, 1)
    return means, quats, scales, opacities, sh


def load_cameras(cams_path, width, height, device):
    """Load camera parameters and return viewmat, K for cam_idx=0."""
    cams = json.loads(Path(cams_path).read_text())
    c = cams[0]
    native_w, native_h = int(c["width"]), int(c["height"])
    R = np.asarray(c["rotation"], dtype=np.float32).T
    p = np.asarray(c["position"], dtype=np.float32)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ p
    K = np.array([[float(c["fx"]) * width / native_w, 0, (width - 1) / 2],
                  [0, float(c["fy"]) * width / native_w, (height - 1) / 2],
                  [0, 0, 1]], dtype=np.float32)
    return torch.tensor(vm, device=device)[None, None], torch.tensor(K, device=device)[None, None]


def make_fixture(scene, max_long_side, device):
    """Create the F0-F4 fixture (projection, culling, tile intersection)."""
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native

    cfg = SCENE_CONFIGS[scene]
    s = min(1.0, max_long_side / max(cfg["native_w"], cfg["native_h"]))
    width, height = round(cfg["native_w"] * s), round(cfg["native_h"] * s)
    tw, th = math.ceil(width / TILE_SIZE), math.ceil(height / TILE_SIZE)

    means, quats, scales, opacities, colors = load_ply_scene(cfg["ply"], device)
    vm, K = load_cameras(cfg["cams"], width, height, device)
    vm, K = vm[:, [0]], K[:, [0]]

    with torch.no_grad():
        ids, _, _ = _cull_gaussians_batched(
            means, quats, scales, vm, K, width, height,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            camera_model="pinhole")
        m, q, s_c, o, c = _gather_visible_native(means, quats, scales, opacities, colors, ids)
        radii, m2d, depths, conics, _ = fully_fused_projection(
            means=m.unsqueeze(0), covars=None, quats=q.unsqueeze(0), scales=s_c.unsqueeze(0),
            viewmats=vm, Ks=K, width=width, height=height,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            packed=False, calc_compensations=False, camera_model="pinhole")
        opa = o[None, None].expand(1, 1, -1).contiguous()
        try:
            _, isect, flat = torch.ops.gsplat.intersect_tile(
                m2d.contiguous(), radii.contiguous(), depths.contiguous(),
                conics.contiguous(), opa.contiguous(), None, None, 1,
                TILE_SIZE, tw, th, True, False, None)
        except RuntimeError:
            _, isect, flat = torch.ops.gsplat.intersect_tile(
                m2d.contiguous(), radii.contiguous(), depths.contiguous(),
                conics.contiguous(), opa.contiguous(), None, None, 1,
                TILE_SIZE, tw, th, True, False)
        offs = isect_offset_encode(isect, 1, tw, th).reshape(1, 1, th, tw)

    fixture = {
        "scene": scene, "width": width, "height": height,
        "tw": tw, "th": th,
        "m2d": m2d[0, 0].contiguous(),      # [N_vis, 2]
        "conics": conics[0, 0].contiguous(), # [N_vis, 3]
        "depths": depths[0, 0].contiguous(), # [N_vis]
        "opacities": o.contiguous(),         # [N_vis]
        "radii": radii[0, 0].contiguous(),   # [N_vis, 2]
        "offs": offs[0, 0].contiguous(),     # [th, tw]
        "flat": flat.contiguous(),           # [n_isects]
        "n_vis": len(o),
        "colors": c.contiguous(),            # [N_vis, K_SH, 3]
        "means": m, "quats": q, "scales": s_c,
        "vm": vm, "K": K,
    }
    return fixture


def make_f5_inputs(fixture, device, vm, K):
    """Prepare the exact tensors needed by rasterize_to_pixels_3dgs."""
    from gsplat.rendering import _maybe_evaluate_sh
    # SH evaluation: _maybe_evaluate_sh(sh_degree, colors[N,K,D], means[1,N,3],
    #   radii[1,1,N,2], viewmats[1,1,4,4], (1,), C, N, True) -> [1, C, N, 3]
    means_b = fixture["means"][None]  # [1, N, 3]
    radii_b = fixture["radii"][None, None]  # [1, 1, N, 2]
    colors_input = fixture["colors"]  # [N, K, 3]
    colors_eval = _maybe_evaluate_sh(
        SH_DEGREE, colors_input, means_b, radii_b, vm,
        (1,), 1, fixture["n_vis"], True)
    colors_eval = colors_eval.contiguous()  # [1, C, N, 3]
    m2d_b = fixture["m2d"][None, None].contiguous()  # [1, 1, N_vis, 2]
    conics_b = fixture["conics"][None, None].contiguous()  # [1, 1, N_vis, 3]
    opa_b = fixture["opacities"][None, None].contiguous()  # [1, 1, N_vis]
    offs_b = fixture["offs"][None, None].contiguous()  # [1, 1, th, tw]
    flat = fixture["flat"].contiguous()  # [n_isects]
    return m2d_b, conics_b, colors_eval, opa_b, offs_b, flat


def time_f5(fixture, device, vm, K, n_warm=20, n_meas=100, reps=5, seed=4200):
    """Time B2 F5 (rasterize_to_pixels_3dgs) with CUDA Events."""
    m2d_b, conics_b, colors_eval, opa_b, offs_b, flat = make_f5_inputs(fixture, device, vm, K)
    w, h = fixture["width"], fixture["height"]

    times = []
    # Warmup
    for _ in range(n_warm):
        torch.ops.gsplat.rasterize_to_pixels_3dgs(
            m2d_b, conics_b, colors_eval, opa_b, None, None,
            w, h, TILE_SIZE, offs_b, flat, False, False)
        torch.cuda.synchronize()

    for rep in range(reps):
        torch.cuda.synchronize()
        for _ in range(n_meas):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.ops.gsplat.rasterize_to_pixels_3dgs(
                m2d_b, conics_b, colors_eval, opa_b, None, None,
                w, h, TILE_SIZE, offs_b, flat, False, False)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        print(f"  F5 timing rep {rep+1}/{reps} done", flush=True)

    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
        "std_ms": float(np.std(arr)),
        "n": len(arr),
    }


def profile_f5_workload(fixture, device):
    """Profile B2 F5 work composition."""
    flat = fixture["flat"].cpu().numpy()
    offs = fixture["offs"].cpu().numpy()
    depths = fixture["depths"].cpu().numpy()
    opacities = fixture["opacities"].cpu().numpy()
    tw, th = fixture["tw"], fixture["th"]
    w, h = fixture["width"], fixture["height"]

    # Per-tile Gaussian counts
    starts = offs.ravel()
    ends = np.append(starts[1:], len(flat))
    tile_counts = np.array([max(0, ends[i] - starts[i]) for i in range(tw * th)])

    # Active tiles (tiles with at least 1 Gaussian)
    active_tiles = int(np.sum(tile_counts > 0))

    # Fine-tile Gaussian entries (total tile-Gaussian pairs)
    n_fine_pairs = int(len(flat))

    # Per-pixel Gaussian evaluations estimate
    # Each tile is TILE_SIZE x TILE_SIZE pixels
    # Each Gaussian in a tile is evaluated at all pixels in the tile
    # but only contributes where its weight > 0
    # We can't get exact per-pixel counts without instrumenting the kernel,
    # but we can estimate from tile-level data

    # Per-tile Gaussian evaluations = tile_counts * TILE_SIZE^2
    total_pixel_gaussian_evals = int(np.sum(tile_counts * TILE_SIZE * TILE_SIZE))

    # Per-pixel statistics
    pixels_per_tile = TILE_SIZE * TILE_SIZE
    # For each tile, each pixel gets tile_counts[tile] Gaussian evaluations
    per_pixel_evals = np.repeat(tile_counts, pixels_per_tile)
    # Only count active pixels (in active tiles)
    per_pixel_active = per_pixel_evals[per_pixel_evals > 0]

    # Alpha acceptance estimate: a Gaussian contributes if alpha > 1/255
    # We can estimate the acceptance rate from opacity statistics
    # but exact alpha depends on the per-pixel Gaussian weight
    # Use opacity > ALPHA_THRESHOLD as a proxy for potential contributors
    ALPHA_THRESHOLD = 1.0 / 255.0
    n_above_threshold = int(np.sum(opacities >= ALPHA_THRESHOLD))
    alpha_accept_rate = n_above_threshold / len(opacities) if len(opacities) > 0 else 0

    # Early termination: pixels where transmittance drops below threshold
    # Can't measure exactly without kernel instrumentation
    # Estimate: pixels in tiles with many Gaussians likely terminate early
    # T_THRESHOLD = 1e-4 means ~8.5 alpha=0.5 Gaussians or ~4 alpha=0.9
    # Rough estimate: if avg tile count > 20, most pixels terminate early

    return {
        "scene": fixture["scene"],
        "width": w, "height": h,
        "n_pixels": w * h,
        "n_visible_gaussians": fixture["n_vis"],
        "n_active_tiles": active_tiles,
        "total_tiles": tw * th,
        "active_tile_fraction": active_tiles / (tw * th),
        "n_fine_tile_gaussian_pairs": n_fine_pairs,
        "total_pixel_gaussian_evaluations": total_pixel_gaussian_evals,
        "per_pixel_evaluations": {
            "mean": float(np.mean(per_pixel_active)) if len(per_pixel_active) > 0 else 0,
            "p50": float(np.percentile(per_pixel_active, 50)) if len(per_pixel_active) > 0 else 0,
            "p90": float(np.percentile(per_pixel_active, 90)) if len(per_pixel_active) > 0 else 0,
            "p99": float(np.percentile(per_pixel_active, 99)) if len(per_pixel_active) > 0 else 0,
            "max": int(np.max(per_pixel_active)) if len(per_pixel_active) > 0 else 0,
        },
        "per_tile_gaussian_count": {
            "mean": float(np.mean(tile_counts)) if active_tiles > 0 else 0,
            "p50": float(np.percentile(tile_counts, 50)),
            "p90": float(np.percentile(tile_counts, 90)),
            "p99": float(np.percentile(tile_counts, 99)),
            "max": int(np.max(tile_counts)),
        },
        "opacity_stats": {
            "mean": float(np.mean(opacities)),
            "min": float(np.min(opacities)),
            "max": float(np.max(opacities)),
            "n_above_alpha_threshold": n_above_threshold,
            "alpha_accept_rate_estimate": alpha_accept_rate,
        },
        "note": "Per-pixel evaluations are estimated from tile-level counts * TILE_SIZE^2. Exact per-pixel alpha acceptance and early termination require kernel instrumentation (not available without modifying the CUDA source). The opacity-based alpha_accept_rate is an upper bound on actual contribution rate.",
    }


def time_existing_higs_raster(scene, max_long_side, device, source, core_so,
                              n_warm=20, n_meas=100, reps=5):
    """Time the existing HiGS macro-tile rasterizer as a performance oracle.

    Uses the experimental extension's gaussian_render_inference_only op.
    NOTE: This produces FP16 output — NOT correctness evidence.
    Only measures raw kernel performance.
    """
    from gsplat.scene.functional.gaussian_inference import pack_gaussian_inference_scene
    try:
        from gsplat.scene.components.gaussian_inference_scene import GaussianInferenceScene
    except ImportError:
        return {"measurable": False, "error": "GaussianInferenceScene not importable"}

    cfg = SCENE_CONFIGS[scene]
    s = min(1.0, max_long_side / max(cfg["native_w"], cfg["native_h"]))
    width, height = round(cfg["native_w"] * s), round(cfg["native_h"] * s)

    means, quats, scales, opacities, colors = load_ply_scene(cfg["ply"], device)
    vm, K = load_cameras(cfg["cams"], width, height, device)

    # Pack into inference format (FP32 -> FP16) via from_gaussian_tensors
    pack_start = torch.cuda.Event(enable_timing=True)
    pack_end = torch.cuda.Event(enable_timing=True)

    try:
        pack_start.record()
        scene_obj = GaussianInferenceScene.from_gaussian_tensors(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, sh_degree=SH_DEGREE, sh_compression="none",
            id=f"oracle_{scene}")
        pack_end.record()
        pack_end.synchronize()
        pack_time_ms = pack_start.elapsed_time(pack_end)
    except Exception as e:
        return {"measurable": False, "error": f"Scene construction failed: {e}"}

    # Try to call the inference render
    try:
        from gsplat.experimental.render.kernels.gaussian_inference_ops import (
            gaussian_render_inference_only)

        # sh_compression_mode: 0=NONE, 1=PACKED_32B, 2=PACKED_16B
        sh_comp_int = int(scene_obj.sh_compression_mode) if scene_obj.sh_compression_mode is not None else 0

        # Warmup
        for _ in range(n_warm):
            try:
                renders, alphas = gaussian_render_inference_only(
                    scene_obj.means_planar, scene_obj.qso_packed,
                    scene_obj.colors_packed,
                    vm[0, 0], K[0, 0], width, height,
                    scene_obj.sh_degree, TILE_SIZE,
                    0.01, 1e10, 0.0, 0.3, sh_comp_int, None)
                torch.cuda.synchronize()
            except Exception as e:
                return {"measurable": False, "error": f"Render call failed: {e}",
                        "pack_time_ms": pack_time_ms}

        times = []
        for rep in range(reps):
            torch.cuda.synchronize()
            for _ in range(n_meas):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                renders, alphas = gaussian_render_inference_only(
                    scene_obj.means_planar, scene_obj.qso_packed,
                    scene_obj.colors_packed,
                    vm[0, 0], K[0, 0], width, height,
                    scene_obj.sh_degree, TILE_SIZE,
                    0.01, 1e10, 0.0, 0.3, sh_comp_int, None)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            print(f"  HiGS raster oracle rep {rep+1}/{reps} done", flush=True)

        arr = np.array(times)
        return {
            "measurable": True,
            "scene": scene,
            "pack_time_ms": pack_time_ms,
            "raster_total": {
                "median_ms": float(np.median(arr)),
                "mean_ms": float(np.mean(arr)),
                "p10_ms": float(np.percentile(arr, 10)),
                "p90_ms": float(np.percentile(arr, 90)),
                "std_ms": float(np.std(arr)),
                "n": len(arr),
            },
            "note": "FP16 output. NOT correctness evidence. Raw kernel performance oracle only.",
        }
    except Exception as e:
        return {"measurable": False, "error": str(e), "pack_time_ms": pack_time_ms}


def compute_break_even(scene, f5_median_ms):
    """Compute break-even and gain targets for Macro-F5."""
    f4 = R2_F4_TIMING[scene]
    b2_f4 = f4["B2_F4_median_ms"]
    macro_f4 = f4["Macro_F4_median_ms"]
    f4_penalty = macro_f4 - b2_f4

    # Break-even: T_macro_F5 < T_B2_F5 - F4_penalty
    max_macro_f5 = f5_median_ms - f4_penalty

    # Required speedup: Macro-F5 must be faster than B2-F5 by at least F4_penalty
    required_speedup_pct = (f4_penalty / f5_median_ms) * 100 if f5_median_ms > 0 else float('inf')

    # Target gains
    # For X% F4+F5 gain: T_macro_F4 + T_macro_F5 < (1 - X/100) * (T_B2_F4 + T_B2_F5)
    # T_macro_F5 < (1 - X/100) * (T_B2_F4 + T_B2_F5) - T_macro_F4
    b2_total = b2_f4 + f5_median_ms
    targets = {}
    for gain_pct in [0, 3, 5, 10]:
        target_total = b2_total * (1 - gain_pct / 100.0)
        target_f5 = target_total - macro_f4
        gain_label = "break_even" if gain_pct == 0 else f"gain_{gain_pct}pct"
        targets[gain_label] = {
            "max_macro_f5_ms": target_f5,
            "required_f5_speedup_pct": ((f5_median_ms - target_f5) / f5_median_ms * 100) if f5_median_ms > 0 and target_f5 > 0 else None,
            "total_f4_f5_target_ms": target_total,
        }

    return {
        "scene": scene,
        "B2_F4_ms": b2_f4,
        "Macro_F4_ms": macro_f4,
        "F4_penalty_ms": f4_penalty,
        "B2_F5_ms": f5_median_ms,
        "B2_F4_F5_total_ms": b2_total,
        "max_macro_F5_for_break_even_ms": max_macro_f5,
        "required_macro_F5_speedup_pct": required_speedup_pct,
        "F4_penalty_pct_of_F5": (f4_penalty / f5_median_ms * 100) if f5_median_ms > 0 else 0,
        "targets": targets,
        "feasibility": "feasible" if max_macro_f5 > 0 else "infeasible",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="/tmp/higs_h3_fwd_1a_source")
    ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--skip-oracle", action="store_true", default=False)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PATH"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    run_id = str(uuid.uuid4())[:12]
    print(f"=== H3-FWD-1B-0: Macro Raster Break-even Oracle ===", flush=True)
    print(f"run_id={run_id} GPU={args.gpu}", flush=True)

    # Bootstrap
    bootstrap(args.source, args.core_so)
    print("Bootstrap complete.", flush=True)

    # ===================== 1. B2 F5 Timing =====================
    print("\n=== 1. B2 F5 (rasterize_to_pixels) Timing ===", flush=True)
    f5_timing = {}
    f5_rows = []
    for scene in ["room", "bicycle", "garden"]:
        print(f"\n--- {scene} ---", flush=True)
        fixture = make_fixture(scene, args.max_long_side, device)
        print(f"  {fixture['width']}x{fixture['height']}, N_vis={fixture['n_vis']}, "
              f"n_isects={len(fixture['flat'])}, tiles={fixture['tw']}x{fixture['th']}", flush=True)
        timing = time_f5(fixture, device, fixture["vm"], fixture["K"],
                         args.warmup, args.measure, args.reps, args.seed)
        f5_timing[scene] = timing
        print(f"  F5 median={timing['median_ms']:.4f}ms mean={timing['mean_ms']:.4f}ms", flush=True)

        row = {"scene": scene, "width": fixture["width"], "height": fixture["height"],
               "n_vis": fixture["n_vis"], "n_isects": len(fixture["flat"]),
               "tw": fixture["tw"], "th": fixture["th"],
               **timing}
        f5_rows.append(row)

        # Save fixture workload
        workload = profile_f5_workload(fixture, device)
        (out_dir / f"b2_f5_workload_{scene}.json").write_text(json.dumps(workload, indent=2))

    # Write F5 timing CSV
    with open(out_dir / "b2_f5_timing.csv", "w", newline="") as f:
        fields = ["scene", "width", "height", "n_vis", "n_isects", "tw", "th",
                  "median_ms", "mean_ms", "p10_ms", "p90_ms", "std_ms", "n"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in f5_rows:
            w.writerow(r)

    # ===================== 2. Break-even Targets =====================
    print("\n=== 2. Break-even Macro-F5 Targets ===", flush=True)
    break_even = {}
    for scene in ["room", "bicycle", "garden"]:
        be = compute_break_even(scene, f5_timing[scene]["median_ms"])
        break_even[scene] = be
        print(f"  {scene}: B2_F5={be['B2_F5_ms']:.4f}ms F4_penalty={be['F4_penalty_ms']:.4f}ms "
              f"max_Macro_F5={be['max_macro_F5_for_break_even_ms']:.4f}ms "
              f"required_speedup={be['required_macro_F5_speedup_pct']:.2f}%", flush=True)
    (out_dir / "break_even_targets.json").write_text(json.dumps(break_even, indent=2))

    # ===================== 3. B2 F5 Workload (combined) =====================
    print("\n=== 3. B2 F5 Workload Composition ===", flush=True)
    workload_all = {}
    for scene in ["room", "bicycle", "garden"]:
        wl_path = out_dir / f"b2_f5_workload_{scene}.json"
        if wl_path.exists():
            workload_all[scene] = json.loads(wl_path.read_text())
            wl = workload_all[scene]
            print(f"  {scene}: {wl['n_fine_tile_gaussian_pairs']} fine pairs, "
                  f"{wl['n_active_tiles']} active tiles, "
                  f"per-pixel mean={wl['per_pixel_evaluations']['mean']:.1f} "
                  f"p90={wl['per_pixel_evaluations']['p90']:.1f}", flush=True)
    (out_dir / "b2_f5_workload.json").write_text(json.dumps(workload_all, indent=2))

    # ===================== 4. Existing HiGS Raster Oracle =====================
    print("\n=== 4. Existing HiGS MacroTileRasterize Oracle ===", flush=True)
    oracle_results = {}
    if not args.skip_oracle:
        for scene in ["room", "bicycle", "garden"]:
            print(f"\n--- {scene} ---", flush=True)
            try:
                result = time_existing_higs_raster(
                    scene, args.max_long_side, device,
                    args.source, args.core_so,
                    args.warmup, args.measure, args.reps)
                oracle_results[scene] = result
                if result.get("measurable"):
                    print(f"  pack={result['pack_time_ms']:.4f}ms "
                          f"raster={result['raster_total']['median_ms']:.4f}ms", flush=True)
                else:
                    print(f"  Not measurable: {result.get('error', 'unknown')}", flush=True)
            except Exception as e:
                oracle_results[scene] = {"measurable": False, "error": str(e)}
                print(f"  Error: {e}", flush=True)
    else:
        print("  Skipped (--skip-oracle)", flush=True)
    (out_dir / "existing_higs_raster_oracle.json").write_text(json.dumps(oracle_results, indent=2))

    # ===================== 5. Opportunity Analysis =====================
    print("\n=== 5. Structural Opportunity Analysis ===", flush=True)
    opportunity = {}
    for scene in ["room", "bicycle", "garden"]:
        be = break_even[scene]
        wl = workload_all.get(scene, {})
        struct = R2_STRUCTURAL[scene]
        compression = struct["N_B2_tile_pairs"] / struct["N_macro_entries"] if struct["N_macro_entries"] > 0 else 0

        # Reducible vs irreducible work
        opportunity[scene] = {
            "scene": scene,
            "representation_compression": compression,
            "n_fine_pairs": struct["N_B2_tile_pairs"],
            "n_macro_entries": struct["N_macro_entries"],
            "B2_F5_ms": be["B2_F5_ms"],
            "F4_penalty_ms": be["F4_penalty_ms"],
            "F4_penalty_pct_of_F5": (be["F4_penalty_ms"] / be["B2_F5_ms"] * 100) if be["B2_F5_ms"] > 0 else 0,
            "required_F5_speedup_pct": be["required_macro_F5_speedup_pct"],
            "max_macro_F5_ms": be["max_macro_F5_for_break_even_ms"],
            "reducible_work": {
                "gaussian_metadata_loads": "Macro raster loads metadata once per macro entry (Gaussian ID + mask) instead of once per fine tile-Gaussian pair. Reduces loads by ~compression factor.",
                "sorted_entry_traversal": "Macro raster traverses N_macro_entries sorted entries instead of N_fine_pairs. Reduces traversal by ~compression factor.",
                "fine_tile_scheduling": "Macro raster skips per-fine-tile scheduling within a macro tile — the 32-bit mask encodes which fine tiles each Gaussian covers.",
                "queue_overhead": "Per-tile queue setup overhead reduced: 1 macro tile queue vs MTW*MTH fine tile queues.",
                "global_memory_traffic": "Reduced: fewer sorted IDs to load, fewer tile offsets to look up. Color/conic data still loaded per unique Gaussian (not per pair).",
            },
            "irreducible_work": {
                "pixel_gaussian_weight_evaluation": "Each pixel must evaluate the Gaussian weight (conic * delta) for every Gaussian covering it. This is the same number of evaluations regardless of representation.",
                "alpha_calculation": "alpha = opac * weight. Same number of calculations.",
                "transmittance_chain": "T *= (1 - alpha). Same number of updates.",
                "color_FMA": "color += T * alpha * rgb. Same number of FMAs.",
                "early_termination_check": "Same check per Gaussian per pixel.",
            },
            "key_insight": f"Macro raster compresses the REPRESENTATION ({compression:.1f}x fewer entries) but NOT the PIXEL COMPUTE (same number of per-pixel Gaussian evaluations). The speedup opportunity is in reduced memory traffic and traversal overhead, NOT in reduced FLOPs.",
        }
        print(f"  {scene}: compression={compression:.1f}x, "
              f"F4_penalty={be['F4_penalty_ms']:.4f}ms ({be['F4_penalty_pct_of_F5']:.1f}% of F5), "
              f"required_speedup={be['required_macro_F5_speedup_pct']:.2f}%", flush=True)
    (out_dir / "opportunity_analysis.json").write_text(json.dumps(opportunity, indent=2))

    # ===================== 6. Decision Gate =====================
    print("\n=== 6. Decision Gate ===", flush=True)
    # Determine STRONG/MARGINAL/WEAK
    # Key metric: required F5 speedup % vs representation compression
    # If required speedup is small relative to the memory traffic reduction opportunity, STRONG
    # If required speedup is moderate, MARGINAL
    # If required speedup exceeds realistic expectations, WEAK

    all_feasible = True
    all_margins = []
    for scene in ["room", "bicycle", "garden"]:
        be = break_even[scene]
        if be["max_macro_F5_for_break_even_ms"] <= 0:
            all_feasible = False
        # Margin = how much faster the macro raster could be vs how much faster it needs to be
        # The compression factor gives an upper bound on memory traffic reduction
        # But pixel compute is irreducible, so actual speedup << compression
        struct = R2_STRUCTURAL[scene]
        compression = struct["N_B2_tile_pairs"] / struct["N_macro_entries"]
        required = be["required_macro_F5_speedup_pct"]
        # A macro raster that eliminates all representation overhead (traversal + metadata loads)
        # could theoretically save up to ~30-50% of F5 time (if F5 is memory-bound)
        # But if pixel compute dominates, the achievable speedup is much less
        margin = compression / required if required > 0 else float('inf')
        all_margins.append(margin)

    avg_margin = np.mean(all_margins)
    # The macro raster's main advantage is fewer global memory loads for sorted IDs and metadata
    # If F5 is compute-bound (pixel math), the speedup opportunity is small
    # If F5 is memory-bound (loading Gaussian data), the speedup opportunity is larger

    # Compare F4 penalty to F5 time
    penalties_pct = [break_even[s]["F4_penalty_pct_of_F5"] for s in ["room", "bicycle", "garden"]]
    avg_penalty_pct = np.mean(penalties_pct)

    if avg_penalty_pct < 10 and all(m > 3 for m in all_margins):
        decision = "RASTER_OPPORTUNITY_STRONG"
    elif avg_penalty_pct < 25 and all(m > 1.5 for m in all_margins):
        decision = "RASTER_OPPORTUNITY_MARGINAL"
    else:
        decision = "RASTER_OPPORTUNITY_WEAK"

    print(f"  Average F4 penalty as % of F5: {avg_penalty_pct:.1f}%", flush=True)
    print(f"  Average margin (compression / required_speedup): {avg_margin:.2f}", flush=True)
    print(f"  Decision: {decision}", flush=True)

    # Add decision to opportunity analysis
    opportunity["decision"] = {
        "gate": decision,
        "avg_f4_penalty_pct_of_f5": float(avg_penalty_pct),
        "avg_margin": float(avg_margin),
        "all_feasible": all_feasible,
        "per_scene": {
            s: {
                "F4_penalty_pct_of_F5": break_even[s]["F4_penalty_pct_of_F5"],
                "required_speedup_pct": break_even[s]["required_macro_F5_speedup_pct"],
                "compression": R2_STRUCTURAL[s]["N_B2_tile_pairs"] / R2_STRUCTURAL[s]["N_macro_entries"],
                "margin": all_margins[i],
            }
            for i, s in enumerate(["room", "bicycle", "garden"])
        },
    }
    (out_dir / "opportunity_analysis.json").write_text(json.dumps(opportunity, indent=2))

    # ===================== Provenance =====================
    provenance = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "gpu": f"cuda:{args.gpu}",
        "source": args.source,
        "core_so": args.core_so,
        "core_so_sha256": hashlib.sha256(Path(args.core_so).read_bytes()).hexdigest() if Path(args.core_so).exists() else "N/A",
        "protocol": {
            "warmup": args.warmup,
            "measurements_per_rep": args.measure,
            "repetitions": args.reps,
            "clock": "CUDA Events",
            "interleaved": False,
            "max_long_side": args.max_long_side,
        },
        "R2_F4_timing_source": "/tmp/h3_fwd_1a_r2/timing_{room,bicycle,garden}.json",
        "R2_structural_source": "/tmp/h3_fwd_1a_r2/{room,bicycle,garden}.json",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print(f"\n=== Done. Decision: {decision} ===", flush=True)
    print(f"\nSummary:", flush=True)
    for scene in ["room", "bicycle", "garden"]:
        be = break_even[scene]
        print(f"  {scene}: B2_F5={be['B2_F5_ms']:.4f}ms F4_penalty={be['F4_penalty_ms']:.4f}ms "
              f"max_Macro_F5={be['max_macro_F5_for_break_even_ms']:.4f}ms "
              f"required_speedup={be['required_macro_F5_speedup_pct']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
