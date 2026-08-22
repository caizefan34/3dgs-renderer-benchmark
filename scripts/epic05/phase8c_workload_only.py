#!/usr/bin/env python3
"""
Phase 8C — Workload Quantification Only (Track D).

Compute detailed tile-Gaussian distribution stats for ALL 6 checkpoints
at both tile16 and tile32. These are the missing workload metrics that
will disambiguate between work-amount and per-work-efficiency hypotheses.

Output: results/epic05/phase8c_workload_stats.json
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

CKPT_MAP = {
    "room_iter5000":  "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt",
    "room_iter10000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter10000.pt",
    "room_iter15000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter15000.pt",
    "room_iter20000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter20000.pt",
    "room_iter25000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter25000.pt",
    "room_iter30000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter30000.pt",
}

CKPT_DIR = REPO_ROOT / "results" / "epic05" / "phase7"
OUT_DIR = REPO_ROOT / "results" / "epic05"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_camera(W=1920, H=1080):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    viewmat = torch.eye(4, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    t = torch.eye(4, device=DEVICE, dtype=DTYPE)
    t[2, 3] = -5.0
    viewmat[0] = t
    K = torch.tensor(
        [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
        device=DEVICE, dtype=DTYPE,
    ).unsqueeze(0)
    return viewmat, K, W, H


def load_checkpoint(path):
    cp = torch.load(path, map_location=DEVICE, weights_only=False)
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
    """Compute detailed workload statistics."""
    with torch.no_grad():
        rendered, alpha, meta = rasterization(
            means=params["xyz"], quats=params["rotations"],
            scales=params["scales"], opacities=params["opacity"],
            colors=params["shs"],
            viewmats=viewmat, Ks=K, width=W, height=H,
            tile_size=tile_size, packed=True, sh_degree=params["sh_degree"],
        )

    tpg = meta["tiles_per_gauss"]  # [nnz]
    tile_w = int(meta["tile_width"])
    tile_h = int(meta["tile_height"])
    total_tiles = tile_w * tile_h
    n_isects = meta["flatten_ids"].shape[0]

    tpg_np = tpg.float().cpu().numpy()

    # Compute gaussians-per-tile from isect_offsets
    isect_offsets = meta["isect_offsets"]  # [1, tile_h, tile_w]
    isect_offsets_np = isect_offsets.cpu().numpy()[0].astype(np.int64).flatten()
    tile_starts = isect_offsets_np.copy()
    tile_ends = np.roll(isect_offsets_np, -1)
    tile_ends[-1] = n_isects
    gpt_np = tile_ends - tile_starts

    empty_tiles = int((gpt_np == 0).sum())
    nnz = tpg_np.shape[0]

    return {
        "total_gaussians": params["xyz"].shape[0],
        "nnz_gaussians_visible": int(nnz),
        "tile_grid": f"{tile_w}x{tile_h}",
        "tile_w": tile_w,
        "tile_h": tile_h,
        "total_tiles": total_tiles,
        "total_intersections_n_isects": int(n_isects),

        # Tiles Per Gaussian distribution
        "tpg_mean": float(tpg_np.mean()),
        "tpg_median": float(np.median(tpg_np)),
        "tpg_std": float(tpg_np.std()),
        "tpg_min": int(tpg_np.min()),
        "tpg_max": int(tpg_np.max()),
        "tpg_p5": float(np.percentile(tpg_np, 5)),
        "tpg_p25": float(np.percentile(tpg_np, 25)),
        "tpg_p75": float(np.percentile(tpg_np, 75)),
        "tpg_p95": float(np.percentile(tpg_np, 95)),
        "tpg_p99": float(np.percentile(tpg_np, 99)),

        # Gaussians Per Tile distribution
        "gpt_mean": float(gpt_np.mean()),
        "gpt_median": float(np.median(gpt_np)),
        "gpt_std": float(gpt_np.std()),
        "gpt_min": int(gpt_np.min()),
        "gpt_max": int(gpt_np.max()),
        "gpt_p5": float(np.percentile(gpt_np, 5)),
        "gpt_p25": float(np.percentile(gpt_np, 25)),
        "gpt_p75": float(np.percentile(gpt_np, 75)),
        "gpt_p95": float(np.percentile(gpt_np, 95)),
        "gpt_p99": float(np.percentile(gpt_np, 99)),
        "empty_tiles": empty_tiles,
        "empty_tile_pct": float(empty_tiles / total_tiles * 100),

        # Derived
        "isects_per_visible_gaussian": float(n_isects / nnz) if nnz > 0 else 0,
        "isects_per_tile": float(n_isects / total_tiles),
    }


def main():
    print("=" * 70)
    print("Phase 8C — Workload Quantification")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    results = {}

    for ckpt_name in sorted(CKPT_MAP.keys()):
        ckpt_path = CKPT_DIR / CKPT_MAP[ckpt_name]
        if not ckpt_path.exists():
            print(f"\n  SKIP {ckpt_name}: not found")
            continue

        print(f"\n{'='*50}")
        print(f"  {ckpt_name}")
        params = load_checkpoint(str(ckpt_path))
        N = params["xyz"].shape[0]
        print(f"  Gaussians: {N:,}")

        viewmat, K, W, H = make_camera()

        entry = {"num_gaussians": N, "sh_degree": params["sh_degree"], "tile_sizes": {}}

        for ts in [16, 32]:
            label = f"tile{ts}"
            print(f"\n  [{label}]")
            wl = compute_workload_stats(params, viewmat, K, W, H, ts)

            print(f"    TPG: mean={wl['tpg_mean']:.1f} median={wl['tpg_median']:.0f} "
                  f"p95={wl['tpg_p95']:.0f} p99={wl['tpg_p99']:.0f} max={wl['tpg_max']}")
            print(f"    GPT: mean={wl['gpt_mean']:.0f} median={wl['gpt_median']:.0f} "
                  f"p95={wl['gpt_p95']:.0f} p99={wl['gpt_p99']:.0f} max={wl['gpt_max']}")
            print(f"    isects={wl['total_intersections_n_isects']/1e6:.1f}M  "
                  f"tiles={wl['total_tiles']}  empty={wl['empty_tiles']} "
                  f"({wl['empty_tile_pct']:.1f}%)")
            print(f"    visible Gs={wl['nnz_gaussians_visible']:,}/{wl['total_gaussians']:,}")

            entry["tile_sizes"][label] = wl

        # Ratios
        r = {}
        w16 = entry["tile_sizes"]["tile16"]
        w32 = entry["tile_sizes"]["tile32"]
        for k in ["total_intersections_n_isects", "tpg_mean", "tpg_median",
                   "gpt_mean", "gpt_median", "gpt_p95", "gpt_p99", "gpt_max",
                   "isects_per_tile"]:
            v16 = w16.get(k, 0)
            v32 = w32.get(k, 0)
            if isinstance(v16, (int, float)) and isinstance(v32, (int, float)) and v32 != 0:
                r[f"{k}_ratio_t16_t32"] = v16 / v32
        entry["ratios"] = r

        print(f"\n  Ratios (t16/t32):")
        for k, v in r.items():
            print(f"    {k}: {v:.2f}×")

        results[ckpt_name] = entry
        torch.cuda.empty_cache()

    # Save
    output = {
        "experiment_id": "phase8c-workload-quantification",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "camera": "synthetic centered (50° FOV, z=-5.0)",
        "resolution": "1920x1080",
        "checkpoints": results,
    }

    out_path = OUT_DIR / "phase8c_workload_stats.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"Saved: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
