#!/usr/bin/env python3
"""
Phase 9A — M2 Forward Correctness: Packed vs Dense pixel comparison.
Standalone version.
"""

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import gc
import os

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import repo modules
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras


def get_first_camera(scene, repo_root, resolution="1080p"):
    """Get first camera for a scene at target resolution."""
    scene_dir = repo_root / "data" / "official" / "mipnerf360" / scene
    cameras_json_path = scene_dir / "cameras.json"
    RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
    target_w, target_h = RESOLUTIONS.get(resolution, (1920, 1080))
    cameras = load_cameras_from_json(str(cameras_json_path))
    if cameras is None or len(cameras) == 0:
        return None
    cam = cameras[0]
    if cam.image_width != target_w or cam.image_height != target_h:
        cam = resize_cameras([cam], target_w, target_h)[0]
    return cam


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M2 Forward Correctness Test")
    parser.add_argument("--scene", default="room", choices=["room", "garden", "bicycle"])
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--resolution", default="1080p")
    args = parser.parse_args()

    device = "cuda"
    results = {
        "experiment_id": f"phase9a_m2_forward_{args.scene}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "date": datetime.now(timezone.utc).isoformat(),
        "config": {"scene": args.scene, "tile_size": args.tile_size, "resolution": args.resolution},
        "tests": [],
    }

    # Load scene PLY data
    scene_dir = REPO_ROOT / "data" / "official" / "mipnerf360" / args.scene
    ply_path = scene_dir / "point_cloud.ply"
    if not ply_path.exists():
        print(f"  PLY not found: {ply_path}")
        results["verdict"] = {"forward_correctness": "BLOCKED", "note": f"PLY not found for {args.scene}"}
        return

    print(f"  Loading PLY: {ply_path}")
    data = load_ply(str(ply_path), device=device)
    means = data["xyz"]
    quats = torch.nn.functional.normalize(data["rotations"], dim=-1)
    scales = torch.exp(data["scales"])
    opacities = torch.sigmoid(data["opacity"]).squeeze(-1) if data["opacity"].dim() > 1 else torch.sigmoid(data["opacity"])
    shs = data["shs"]
    print(f"  Gaussians: {means.shape[0]:,}, SH: {shs.shape}")

    # Camera
    cam = get_first_camera(args.scene, REPO_ROOT, args.resolution)
    if cam is None:
        results["verdict"] = {"forward_correctness": "BLOCKED", "note": "No camera"}
        return

    viewmat = cam.viewmatrix.unsqueeze(0).to(device)
    K = cam.K.unsqueeze(0).to(device)
    H, W = cam.image_height, cam.image_width
    results["config"]["image_size"] = f"{H}x{W}"

    # Run both modes
    from gsplat import rasterization

    # Warmup
    for _ in range(3):
        rasterization(means=means, quats=quats, scales=scales, opacities=opacities,
                      colors=shs, viewmats=viewmat, Ks=K, width=W, height=H,
                      tile_size=args.tile_size, packed=True, sh_degree=3, render_mode="RGB")

    torch.cuda.synchronize()

    # Packed
    ev1 = torch.cuda.Event(enable_timing=True)
    ev2 = torch.cuda.Event(enable_timing=True)
    ev1.record()
    r_pack, a_pack, meta_pack = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=shs, viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=args.tile_size, packed=True, sh_degree=3, render_mode="RGB")
    ev2.record()
    torch.cuda.synchronize()
    t_pack = ev1.elapsed_time(ev2)
    img_pack = r_pack[0].clamp(0, 1)
    nnz = int(meta_pack["gaussian_ids"].shape[0])

    # Dense
    ev1.record()
    r_dense, a_dense, meta_dense = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=shs, viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=args.tile_size, packed=False, sh_degree=3, render_mode="RGB")
    ev2.record()
    torch.cuda.synchronize()
    t_dense = ev1.elapsed_time(ev2)
    img_dense = r_dense[0].clamp(0, 1)

    # Compare
    diff = (img_pack - img_dense).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_err = (diff / (img_dense.abs() + 1e-8)).mean().item()

    print(f"\n  {'='*60}")
    print(f"  M2 Forward Correctness — {args.scene}")
    print(f"{'='*60}")
    print(f"  Packed: {t_pack:.2f}ms, nnz={nnz:,}/{means.shape[0]:,} ({100.0*nnz/means.shape[0]:.1f}%)")
    print(f"  Dense:  {t_dense:.2f}ms")
    print(f"  Max abs diff:  {max_diff:.10f}")
    print(f"  Mean abs diff: {mean_diff:.10f}")
    print(f"  Relative err:  {rel_err:.10f}")

    fwd_ok = max_diff < 0.01  # tolerance < 1% of pixel range
    print(f"  Forward equivalence: {'PASS' if fwd_ok else 'FAIL'}")

    results["tests"].append({
        "test_name": f"sfm_init_{args.scene}",
        "n_gaussians": means.shape[0],
        "packed_nnz": nnz,
        "visible_pct": round(100.0 * nnz / means.shape[0], 1),
        "max_abs_diff": float(f"{max_diff:.10f}"),
        "mean_abs_diff": float(f"{mean_diff:.10f}"),
        "relative_error": float(f"{rel_err:.10f}"),
        "packed_time_ms": round(t_pack, 2),
        "dense_time_ms": round(t_dense, 2),
        "speedup_packed_vs_dense": round(t_dense / max(t_pack, 0.01), 3),
        "forward_equivalence": "PASS" if fwd_ok else "FAIL",
    })

    results["verdict"] = {
        "forward_correctness": "SUPPORTED" if fwd_ok else "FAILED",
        "note": f"Packed vs dense pixel comparison on {args.scene}. Max abs diff={max_diff:.2e}.",
    }

    # Save
    out_dir = REPO_ROOT / "results" / "epic05" / "phase9a"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"m2_forward_{args.scene}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
