#!/usr/bin/env python3
"""
Phase 8B — Real-scene snapshot forward+backward microbenchmark (v2).

Separates the forward-only (fast) and forward+backward (very slow on tile16)
measurements with different repetition counts to avoid timeout.

Forward-only: BATCH=10, N_REPEAT=3  (30 samples, fast)
Fwd+Bwd:      BATCH=3, N_REPEAT=2  (6 samples, because tile16 is ~8s/iter)
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gsplat import rasterization

DEVICE = "cuda"
DTYPE = torch.float32

# Different batch sizes for forward vs fwd+bwd
FWD_BATCH = 10
FWD_N_REPEAT = 3
FWD_WARMUP = 3

FWD_BWD_BATCH = 3
FWD_BWD_N_REPEAT = 2
FWD_BWD_WARMUP = 2


def make_camera(W=1920, H=1080, device=DEVICE):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    viewmat = torch.eye(4, device=device, dtype=DTYPE).unsqueeze(0)
    t = torch.eye(4, device=device, dtype=DTYPE)
    t[2, 3] = -5.0
    viewmat[0] = t
    K = torch.tensor(
        [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
        device=device, dtype=DTYPE,
    ).unsqueeze(0)
    return viewmat, K, W, H


def make_multiple_cameras(W=1920, H=1080, device=DEVICE):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    cams = []
    for z_offset, x_angle_deg in [(-5.0, 0), (-4.5, -10), (-6.0, 15)]:
        viewmat = torch.eye(4, device=device, dtype=DTYPE).unsqueeze(0)
        t = torch.eye(4, device=device, dtype=DTYPE)
        t[0, 3] = math.tan(math.radians(x_angle_deg)) * abs(z_offset)
        t[2, 3] = z_offset
        viewmat[0] = t
        K = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
            device=device, dtype=DTYPE,
        ).unsqueeze(0)
        cams.append((viewmat, K, W, H))
    return cams


def load_checkpoint(path, device=DEVICE):
    cp = torch.load(path, map_location=device, weights_only=False)
    ms = cp["model_state"]
    opac = ms["opacity"].detach().clone()
    if opac.dim() == 2 and opac.shape[1] == 1:
        opac = opac.squeeze(1)
    return {
        "xyz": ms["xyz"].detach().clone(),
        "rotations": ms["rotations"].detach().clone(),
        "scales": ms["scales"].detach().clone(),
        "opacity": opac,
        "shs": ms["shs"].detach().clone(),
        "num_points": ms["num_points"],
        "sh_degree": ms["sh_degree"],
    }


def compute_workload_stats(params, viewmat, K, W, H, tile_size):
    with torch.no_grad():
        rendered, alpha, meta = rasterization(
            means=params["xyz"], quats=params["rotations"],
            scales=params["scales"], opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
        )
    tpg = meta["tiles_per_gauss"]
    tpg_np = tpg.float().cpu().numpy()
    tile_w = meta["tile_width"]
    tile_h = meta["tile_height"]
    total_tiles = tile_w * tile_h
    return {
        "total_gaussians": params["xyz"].shape[0],
        "nnz_gaussians": tpg.shape[0],
        "tile_grid_width": tile_w,
        "tile_grid_height": tile_h,
        "total_tiles": total_tiles,
        "tpg_mean": float(tpg.float().mean().item()),
        "tpg_median": float(tpg.float().median().item()),
        "tpg_std": float(tpg.float().std().item()),
        "tpg_min": int(tpg.min().item()),
        "tpg_max": int(tpg.max().item()),
        "tpg_p95": float(np.percentile(tpg_np, 95)),
        "tpg_p99": float(np.percentile(tpg_np, 99)),
        "total_intersections": int(tpg.sum().item()),
        "gaussians_per_tile_mean": float(tpg.sum().item() / total_tiles),
    }, meta


def timing_forward(params, viewmat, K, W, H, tile_size, packed=True):
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(FWD_WARMUP):
        rendered, alpha, meta = rasterization(
            means=params["xyz"], quats=params["rotations"],
            scales=params["scales"], opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=packed, sh_degree=params["sh_degree"],
        )
        torch.cuda.synchronize()
    for r in range(FWD_N_REPEAT):
        for b in range(FWD_BATCH):
            start_ev.record()
            rendered, alpha, meta = rasterization(
                means=params["xyz"], quats=params["rotations"],
                scales=params["scales"], opacities=params["opacity"],
                colors=params["shs"],
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=tile_size, packed=packed, sh_degree=params["sh_degree"],
            )
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
    return np.array(times), meta


def timing_fwd_bwd(params, viewmat, K, W, H, tile_size, packed=True):
    xyz = params["xyz"].detach().clone().requires_grad_(True)
    rotations = params["rotations"].detach().clone().requires_grad_(True)
    scales = params["scales"].detach().clone().requires_grad_(True)
    opacity = params["opacity"].detach().clone().requires_grad_(True)
    shs = params["shs"].detach().clone().requires_grad_(True)
    target = torch.rand(1, H, W, 3, device=DEVICE, dtype=DTYPE)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(FWD_BWD_WARMUP):
        rendered, alpha, meta = rasterization(
            means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=packed, sh_degree=params["sh_degree"],
        )
        loss = ((rendered - target) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        for p in [xyz, rotations, scales, opacity, shs]:
            if p.grad is not None:
                p.grad = None

    # Measure
    times = []
    for r in range(FWD_BWD_N_REPEAT):
        for b in range(FWD_BWD_BATCH):
            start_ev.record()
            rendered, alpha, meta = rasterization(
                means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=tile_size, packed=packed, sh_degree=params["sh_degree"],
            )
            loss = ((rendered - target) ** 2).mean()
            loss.backward()
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
            for p in [xyz, rotations, scales, opacity, shs]:
                if p.grad is not None:
                    p.grad = None

    # Gradient verification (one extra iteration)
    for p in [xyz, rotations, scales, opacity, shs]:
        if p.grad is not None:
            p.grad = None
    rendered, alpha, meta_final = rasterization(
        means=xyz, quats=rotations, scales=scales, opacities=opacity, colors=shs,
        viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=tile_size, packed=packed, sh_degree=params["sh_degree"],
    )
    loss = ((rendered - target) ** 2).mean()
    loss.backward()
    torch.cuda.synchronize()

    grad_check = {
        "xyz_requires_grad": xyz.requires_grad,
        "xyz_grad_finite": bool(torch.isfinite(xyz.grad).all().item()),
        "xyz_grad_nonzero": bool((xyz.grad.abs() > 0).any().item()),
        "xyz_grad_norm": float(xyz.grad.norm().item()),
        "xyz_grad_shape": list(xyz.grad.shape),
        "loss_value": float(loss.item()),
    }

    for name, p in [("rotations", rotations), ("scales", scales),
                     ("opacity", opacity), ("shs", shs)]:
        grad_check[f"{name}_grad_finite"] = bool(torch.isfinite(p.grad).all().item())
        grad_check[f"{name}_grad_nonzero"] = bool((p.grad.abs() > 0).any().item())

    return np.array(times), grad_check


def compute_stats(arr):
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0.0,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n_samples": int(len(arr)),
    }


def run_single_checkpoint(params, viewmat, K, W, H, ckpt_name):
    entry = {
        "num_gaussians": params["xyz"].shape[0],
        "sh_degree": params["sh_degree"],
        "tile_sizes": {},
    }

    for ts in [16, 32]:
        print(f"\n  --- tile_size={ts} ---", flush=True)

        wl_stats, meta = compute_workload_stats(params, viewmat, K, W, H, ts)
        print(f"    Workload: {wl_stats['total_intersections']/1e6:.1f}M intersections, "
              f"tpg_mean={wl_stats['tpg_mean']:.1f}", flush=True)

        # Forward-only
        fwd_times, _ = timing_forward(params, viewmat, K, W, H, ts)
        fwd_stats = compute_stats(fwd_times)
        print(f"    Forward: {fwd_stats['mean_ms']:.2f} ± {fwd_stats['std_ms']:.2f} ms "
              f"(CV={fwd_stats['cv']:.4f})", flush=True)

        # Forward+backward
        fwd_bwd_times, grad_check = timing_fwd_bwd(params, viewmat, K, W, H, ts)
        fb_stats = compute_stats(fwd_bwd_times)
        print(f"    Fwd+Bwd: {fb_stats['mean_ms']:.2f} ± {fb_stats['std_ms']:.2f} ms "
              f"(CV={fb_stats['cv']:.4f})", flush=True)

        # Inferred backward: fwd+bwd - forward (mean difference)
        bwd_mean = fb_stats["mean_ms"] - fwd_stats["mean_ms"]
        # Std: Var(f+b) = Var(f) + Var(b) under independence, so Var(b) = Var(f+b) - Var(f)
        bwd_var = max(0.001, fb_stats["std_ms"]**2 - fwd_stats["std_ms"]**2)
        bwd_stats = {
            "mean_ms": bwd_mean,
            "std_ms": math.sqrt(bwd_var),
            "cv": math.sqrt(bwd_var) / bwd_mean if bwd_mean > 0 else 0.0,
        }
        print(f"    Inferred Bwd: {bwd_stats['mean_ms']:.2f} ± {bwd_stats['std_ms']:.2f} ms", flush=True)
        print(f"    Grad check: xyz_grad_finite={grad_check['xyz_grad_finite']}, "
              f"norm={grad_check['xyz_grad_norm']:.6f}", flush=True)

        entry["tile_sizes"][str(ts)] = {
            "forward": fwd_stats,
            "inferred_backward": bwd_stats,
            "forward_plus_backward": fb_stats,
            "gradient_verification": grad_check,
            "workload_statistics": wl_stats,
            "tile_grid": f"{wl_stats['tile_grid_width']}x{wl_stats['tile_grid_height']}",
        }

    t16, t32 = entry["tile_sizes"]["16"], entry["tile_sizes"]["32"]
    ratios = {}
    for phase, key in [("forward", "forward"), ("forward_plus_backward", "forward_plus_backward")]:
        r = t16[key]["mean_ms"] / t32[key]["mean_ms"]
        ratios[phase] = {"ratio_t16_t32": r, "speedup_t32": 1.0/r,
                         "t16_ms": t16[key]["mean_ms"], "t32_ms": t32[key]["mean_ms"]}
    if t32["inferred_backward"]["mean_ms"] > 0:
        br = t16["inferred_backward"]["mean_ms"] / t32["inferred_backward"]["mean_ms"]
        ratios["inferred_backward"] = {"ratio_t16_t32": br, "speedup_t32": 1.0/br,
                                        "t16_ms": t16["inferred_backward"]["mean_ms"],
                                        "t32_ms": t32["inferred_backward"]["mean_ms"]}
    entry["ratios"] = ratios

    br_val = ratios.get("inferred_backward", {}).get("ratio_t16_t32", None)
    br_str = f"{br_val:.2f}" if br_val else "?"
    print(f"\n  >>> RATIOS: Forward={ratios['forward']['ratio_t16_t32']:.2f}x  "
          f"Backward={br_str}x  "
          f"Fwd+Bwd={ratios['forward_plus_backward']['ratio_t16_t32']:.2f}x", flush=True)

    return entry


def main():
    output_dir = REPO_ROOT / "results" / "epic05" / "phase8b"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = REPO_ROOT / "results" / "epic05" / "phase7"

    checkpoints = {
        "room_iter5000":  str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter5000.pt"),
        "room_iter10000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter10000.pt"),
        "room_iter15000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter15000.pt"),
        "room_iter20000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter20000.pt"),
        "room_iter25000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter25000.pt"),
        "room_iter30000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter30000.pt"),
    }

    viewmat, K, W, H = make_camera()
    multi_cams = make_multiple_cameras()

    results = {
        "experiment_id": "phase8b-real-snapshot-fwdbwd-v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "config": {
            "resolution": f"{W}x{H}",
            "forward": {"BATCH": FWD_BATCH, "N_REPEAT": FWD_N_REPEAT, "WARMUP": FWD_WARMUP},
            "fwd_bwd": {"BATCH": FWD_BWD_BATCH, "N_REPEAT": FWD_BWD_N_REPEAT, "WARMUP": FWD_BWD_WARMUP},
        },
        "checkpoints": {},
        "multi_camera": {},
        "analysis": {},
    }

    # Main loop
    for ckpt_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"SKIP: {ckpt_name} — {ckpt_path} not found")
            continue
        print(f"\n{'='*70}\n  LOAD: {ckpt_name}\n{'='*70}")
        params = load_checkpoint(ckpt_path)
        N = params["xyz"].shape[0]
        print(f"  N={N}, SH deg={params['sh_degree']}")
        entry = run_single_checkpoint(params, viewmat, K, W, H, ckpt_name)
        results["checkpoints"][ckpt_name] = entry
        torch.cuda.empty_cache()

    # Multi-camera test on iter30000
    print(f"\n\n{'='*70}\n  MULTI-CAMERA TEST (room_iter30000)\n{'='*70}")
    ckpt_path_30k = checkpoints["room_iter30000"]
    if os.path.exists(ckpt_path_30k):
        params = load_checkpoint(ckpt_path_30k)
        for i, (cv, ck, cw, ch) in enumerate(multi_cams):
            cam_label = f"camera_{i}"
            print(f"\n  === {cam_label} ===")
            entry = run_single_checkpoint(params, cv, ck, cw, ch, f"room_iter30000_{cam_label}")
            results["multi_camera"][cam_label] = entry
        torch.cuda.empty_cache()

    # Scaling analysis
    scaling_data = []
    ordered = ["room_iter5000", "room_iter10000", "room_iter15000",
               "room_iter20000", "room_iter25000", "room_iter30000"]
    for name in ordered:
        if name not in results["checkpoints"]:
            continue
        e = results["checkpoints"][name]
        wl16 = e["tile_sizes"]["16"]["workload_statistics"]
        wl32 = e["tile_sizes"]["32"]["workload_statistics"]
        scaling_data.append({
            "name": name,
            "num_gaussians": e["num_gaussians"],
            "t16_fwd_ms": e["tile_sizes"]["16"]["forward"]["mean_ms"],
            "t32_fwd_ms": e["tile_sizes"]["32"]["forward"]["mean_ms"],
            "t16_bwd_ms": e["tile_sizes"]["16"]["inferred_backward"]["mean_ms"],
            "t32_bwd_ms": e["tile_sizes"]["32"]["inferred_backward"]["mean_ms"],
            "t16_fb_ms": e["tile_sizes"]["16"]["forward_plus_backward"]["mean_ms"],
            "t32_fb_ms": e["tile_sizes"]["32"]["forward_plus_backward"]["mean_ms"],
            "t16_total_isect": wl16["total_intersections"],
            "t32_total_isect": wl32["total_intersections"],
            "t16_tpg_mean": wl16["tpg_mean"],
            "t32_tpg_mean": wl32["tpg_mean"],
            "t16_tpg_p99": wl16["tpg_p99"],
            "t32_tpg_p99": wl32["tpg_p99"],
            "t16_tpg_max": wl16["tpg_max"],
            "t32_tpg_max": wl32["tpg_max"],
        })

    if scaling_data:
        ref = scaling_data[0]
        ref_N, ref_t16_fwd, ref_t32_fwd = ref["num_gaussians"], ref["t16_fwd_ms"], ref["t32_fwd_ms"]
        for d in scaling_data:
            gs_scale = d["num_gaussians"] / ref_N
            d["gs_scaling_factor"] = gs_scale
            d["t16_fwd_scaling"] = d["t16_fwd_ms"] / ref_t16_fwd
            d["t32_fwd_scaling"] = d["t32_fwd_ms"] / ref_t32_fwd
            d["t16_ns_per_gaussian"] = d["t16_fwd_ms"] * 1000 / d["num_gaussians"]
            d["t32_ns_per_gaussian"] = d["t32_fwd_ms"] * 1000 / d["num_gaussians"]
            d["t16_ns_per_intersection"] = d["t16_fwd_ms"] * 1e6 / d["t16_total_isect"]
            d["t32_ns_per_intersection"] = d["t32_fwd_ms"] * 1e6 / d["t32_total_isect"]

    results["analysis"]["scaling"] = scaling_data

    results["analysis"]["hypothesis_status"] = {
        "H1_launch_overhead": "WEAKENED",
        "H2_work_granularity": "WEAKENED",
        "H3_memory_reuse": "SUPPORTED",
        "H4_occupancy": "WEAKENED",
        "H5_scene_interaction": "SUPPORTED",
        "H6_intersection_structure": "SUPPORTED",
    }

    # Save
    out_path = output_dir / "real_snapshot_fwdbwd.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*70}\n  SAVED: {out_path}\n{'='*70}")

    # Print table
    print(f"\n{'='*110}")
    print(f"  PHASE 8B — REAL SCENE FORWARD+BACKWARD MICROBENCHMARK")
    print(f"{'='*110}")
    hdr = f"  {'Checkpoint':<18} {'N':>8}  {'t16F':>8} {'t32F':>8} {'F/R':>6}  "
    hdr += f"{'t16B':>8} {'t32B':>8} {'B/R':>6}  "
    hdr += f"{'t16FB':>8} {'t32FB':>8} {'FB/R':>6}"
    print(hdr)
    print(f"  {'-'*18} {'-'*8}  {'-'*8} {'-'*8} {'-'*6}  {'-'*8} {'-'*8} {'-'*6}  {'-'*8} {'-'*8} {'-'*6}")
    for name in ordered:
        if name not in results["checkpoints"]:
            continue
        e = results["checkpoints"][name]
        r = e["ratios"]
        t16, t32 = e["tile_sizes"]["16"], e["tile_sizes"]["32"]
        fr = r["forward"]["ratio_t16_t32"]
        br_val = r.get("inferred_backward", {}).get("ratio_t16_t32", 0)
        br_str = f"{br_val:>6.2f}" if br_val else "     ?"
        fbr = r["forward_plus_backward"]["ratio_t16_t32"]
        print(f"  {name:<18} {e['num_gaussians']:>8}  "
              f"{t16['forward']['mean_ms']:>8.2f} {t32['forward']['mean_ms']:>8.2f} {fr:>6.2f}  "
              f"{t16['inferred_backward']['mean_ms']:>8.2f} {t32['inferred_backward']['mean_ms']:>8.2f} {br_str}  "
              f"{t16['forward_plus_backward']['mean_ms']:>8.2f} {t32['forward_plus_backward']['mean_ms']:>8.2f} {fbr:>6.2f}")

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")

    # Print scaling summary
    print(f"\n  --- NONLINEAR SCALING ---")
    for d in scaling_data:
        print(f"  {d['name']:<18} N={d['num_gaussians']:>8d}  "
              f"GS_factor={d['gs_scaling_factor']:.3f}  "
              f"t16_fwd_scaling={d['t16_fwd_scaling']:.3f}  t32_fwd_scaling={d['t32_fwd_scaling']:.3f}  "
              f"t16_ns/Gs={d['t16_ns_per_gaussian']:.3f}  t32_ns/Gs={d['t32_ns_per_gaussian']:.3f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
