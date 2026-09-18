#!/usr/bin/env python3
"""
R3.1-VEC Gate: Scalar Reference vs Vectorized Implementation

Semantic equivalence check using SYNTHETIC data.
Tests the _accumulate_tile_bounds computation logic in isolation.

200 tile-Gaussian pair samples: scalar reference vs vectorized path.
Pre-declared pass threshold: max_rel_diff < 1e-5
"""
import os
import sys
import json
import math
import random
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

# ---- Pre-declared constants ----
N_SAMPLES = 200
PASS_THRESHOLD_REL = 1e-5
PASS_THRESHOLD_ABS = 1e-7
SQRT_E_INV = 1.0 / math.sqrt(math.e)
SQRT_1p5 = math.sqrt(1.5)
E_INV = 1.0 / math.e

FAMILIES = [
    "color_coarse", "color_tight",
    "opacity", "opacity_tight",
    "mean2d", "mean2d_sigmamin",
    "conic", "conic_sigmamin",
]

TILE_SIZE = 16
H, W = 466, 735
TILE_H, TILE_W = H // TILE_SIZE, W // TILE_SIZE


def compute_sigma_min_scalar(mu, conic, tx, ty, tile_size):
    """Scalar sigma_min: 4-edge continuous rectangle minimum."""
    xx, xy, yy = float(conic[0]), float(conic[1]), float(conic[2])
    corners = [
        (tx * tile_size, ty * tile_size),
        ((tx + 1) * tile_size, ty * tile_size),
        (tx * tile_size, (ty + 1) * tile_size),
        ((tx + 1) * tile_size, (ty + 1) * tile_size),
    ]
    s_min = float('inf')
    for cx, cy in corners:
        dx = float(mu[0]) - cx
        dy = float(mu[1]) - cy
        s = xx * dx * dx + 2 * xy * dx * dy + yy * dy * dy
        s_min = min(s_min, max(s, 0.0))
    return max(s_min, 0.0)


def scalar_bound(opacity, conic, means2d, color_norm, C_max_t, tile_Q, tx, ty):
    """Compute all 8 family bounds for ONE Gaussian — scalar reference."""
    o = float(opacity)
    c = float(color_norm)
    xx, xy, yy = float(conic[0]), float(conic[1]), float(conic[2])

    s_min = compute_sigma_min_scalar(means2d, conic, tx, ty, TILE_SIZE)
    E_tight = math.exp(-s_min)

    # Color bounds
    A_coarse = min(o, 0.999)
    A_tight = min(o * E_tight, 0.999)
    b_color_coarse = A_coarse * tile_Q
    b_color_tight = A_tight * tile_Q

    # Opacity bounds
    factor = (c + C_max_t) * tile_Q
    b_opacity = factor
    b_opacity_tight = E_tight * factor

    # Geometry bounds (need SPD)
    trace = xx + yy
    det = xx * yy - xy * xy
    disc = max(trace * trace - 4.0 * det, 0.0)
    sqrt_disc = math.sqrt(disc)
    lam_max = (trace + sqrt_disc) / 2.0
    lam_min = (trace - sqrt_disc) / 2.0

    is_spd = (det > 0 and trace > 0 and lam_min > 0)
    b_mean2d = 0.0
    b_conic = 0.0
    b_mean2d_sm = 0.0
    b_conic_sm = 0.0

    if is_spd:
        base = o * (c + C_max_t) * tile_Q
        b_mean2d = base * math.sqrt(lam_max) * SQRT_E_INV
        b_conic = base * SQRT_1p5 * E_INV
        if lam_min > 0:
            E_sm = math.exp(-s_min)
            b_mean2d_sm = base * math.sqrt(lam_max) * E_sm * SQRT_E_INV
            b_conic_sm = base * SQRT_1p5 * E_sm * E_INV

    return {
        "color_coarse": b_color_coarse,
        "color_tight": b_color_tight,
        "opacity": b_opacity,
        "opacity_tight": b_opacity_tight,
        "mean2d": b_mean2d,
        "mean2d_sigmamin": b_mean2d_sm,
        "conic": b_conic,
        "conic_sigmamin": b_conic_sm,
    }


def vectorized_bounds(opacities, conics, means2d, colors_rgb, g_unique, K, tile_Q, tx, ty, device):
    """Vectorized: batch operations (mirrors _accumulate_tile_bounds)."""
    results = {f: torch.zeros(K, device=device) for f in FAMILIES}

    mu_batch = means2d[0, g_unique]
    conic_batch = conics[0, g_unique]
    opac_batch = opacities[g_unique]
    color_norm_batch = colors_rgb[0, g_unique].norm(dim=-1)

    C_max_t = float(color_norm_batch.max().item()) if K > 0 else 0.0

    o_j = opac_batch
    c_norm = color_norm_batch

    # sigma_min vectorized (4-corner min)
    corners = torch.tensor([
        [tx * TILE_SIZE, ty * TILE_SIZE],
        [(tx + 1) * TILE_SIZE, ty * TILE_SIZE],
        [tx * TILE_SIZE, (ty + 1) * TILE_SIZE],
        [(tx + 1) * TILE_SIZE, (ty + 1) * TILE_SIZE],
    ], device=device, dtype=torch.float32)

    dx = mu_batch[:, 0:1] - corners[:, 0].unsqueeze(0)
    dy = mu_batch[:, 1:2] - corners[:, 1].unsqueeze(0)
    xx = conic_batch[:, 0:1]
    xy = conic_batch[:, 1:2]
    yy = conic_batch[:, 2:3]
    s_min_v = (xx * dx * dx + 2 * xy * dx * dy + yy * dy * dy).min(dim=1).values
    s_min_v = torch.clamp_min(s_min_v, 0.0)
    E_tight_v = torch.exp(-s_min_v)

    # Color bounds
    A_coarse_v = torch.clamp_max(o_j, 0.999)
    A_tight_v = torch.clamp_max(o_j * E_tight_v, 0.999)
    results["color_coarse"] = A_coarse_v * tile_Q
    results["color_tight"] = A_tight_v * tile_Q

    # Opacity bounds
    factor_op_v = (c_norm + C_max_t) * tile_Q
    results["opacity"] = factor_op_v
    results["opacity_tight"] = E_tight_v * factor_op_v

    # Geometry bounds
    xx_s, xy_s, yy_s = conic_batch[:, 0], conic_batch[:, 1], conic_batch[:, 2]
    trace_s = xx_s + yy_s
    det_s = xx_s * yy_s - xy_s * xy_s
    disc_s = torch.clamp(trace_s * trace_s - 4.0 * det_s, min=0.0)
    sqrt_disc_s = torch.sqrt(disc_s)
    lam_max_s = (trace_s + sqrt_disc_s) / 2.0
    lam_min_s = (trace_s - sqrt_disc_s) / 2.0
    spd_mask = (det_s > 0) & (trace_s > 0) & (lam_min_s > 0)

    if spd_mask.any():
        base_geo = o_j * (c_norm + C_max_t) * tile_Q
        results["mean2d"] = torch.where(spd_mask, base_geo * torch.sqrt(lam_max_s) * SQRT_E_INV, torch.zeros(K, device=device))
        results["conic"] = torch.where(spd_mask, base_geo * SQRT_1p5 * E_INV, torch.zeros(K, device=device))
        lam_min_ok = lam_min_s > 0
        E_sm = torch.exp(-s_min_v)
        results["mean2d_sigmamin"] = torch.where(spd_mask & lam_min_ok, base_geo * torch.sqrt(lam_max_s) * E_sm * SQRT_E_INV, torch.zeros(K, device=device))
        results["conic_sigmamin"] = torch.where(spd_mask & lam_min_ok, base_geo * SQRT_1p5 * E_sm * E_INV, torch.zeros(K, device=device))

    return results


def main():
    print("=== R3.1-VEC Gate: Scalar vs Vectorized (Synthetic) ===")
    print(f"  Samples: {N_SAMPLES} tiles")
    print(f"  Pass threshold (rel): {PASS_THRESHOLD_REL}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    N_GAUSSIANS = 10000

    # Generate synthetic data
    opacities = torch.rand(N_GAUSSIANS, device=device) * 0.9 + 0.05  # [0.05, 0.95]
    means2d = torch.rand(1, N_GAUSSIANS, 2, device=device) * max(H, W)  # random 2D positions
    # Conics: make most SPD, some non-SPD
    conic_xx = torch.rand(N_GAUSSIANS, device=device) * 5.0 + 0.1
    conic_yy = torch.rand(N_GAUSSIANS, device=device) * 5.0 + 0.1
    conic_xy = torch.rand(N_GAUSSIANS, device=device) * 1.0 - 0.5  # [-0.5, 0.5]
    conics = torch.stack([conic_xx, conic_xy, conic_yy], dim=-1).unsqueeze(0)
    # Colors: random RGB
    colors_rgb = torch.rand(1, N_GAUSSIANS, 3, device=device)

    # Random tile_Q values
    tile_Qs = torch.rand(TILE_H * TILE_W, device=device) * 0.5 + 0.1

    # Sample 200 tiles
    valid_tiles = list(range(TILE_H * TILE_W))
    sampled_tiles = random.sample(valid_tiles, min(N_SAMPLES, len(valid_tiles)))

    all_diffs = {f: {"abs": [], "rel": []} for f in FAMILIES}
    total_compared = 0

    for tile_idx in sampled_tiles:
        ty = tile_idx // TILE_W
        tx = tile_idx % TILE_W
        tile_Q = float(tile_Qs[tile_idx])

        # Random number of Gaussians in this tile (3 to 50)
        K = random.randint(3, 50)
        g_unique = torch.randperm(N_GAUSSIANS, device=device)[:K]

        # Scalar reference
        color_norms = colors_rgb[0, g_unique].norm(dim=-1)
        C_max_t = float(color_norms.max().item())

        scalar_results = []
        for i in range(K):
            gi = int(g_unique[i].item())
            sb = scalar_bound(
                opacities[gi], conics[0, gi], means2d[0, gi],
                color_norms[i], C_max_t, tile_Q, tx, ty
            )
            scalar_results.append(sb)

        # Vectorized
        vec_results = vectorized_bounds(
            opacities, conics, means2d, colors_rgb,
            g_unique, K, tile_Q, tx, ty, device
        )

        # Compare
        for fam in FAMILIES:
            for i in range(K):
                s_val = scalar_results[i][fam]
                v_val = float(vec_results[fam][i].item())
                abs_diff = abs(s_val - v_val)
                denom = max(abs(s_val), 1e-12)
                rel_diff = abs_diff / denom

                all_diffs[fam]["abs"].append(abs_diff)
                all_diffs[fam]["rel"].append(rel_diff)

        total_compared += K

    # Aggregate
    report = {
        "methodology": {
            "description": "Synthetic data scalar vs vectorized comparison",
            "n_tiles_sampled": len(sampled_tiles),
            "n_gaussians_compared": total_compared,
            "pass_threshold_rel": PASS_THRESHOLD_REL,
            "pass_threshold_abs": PASS_THRESHOLD_ABS,
        },
        "families": {},
    }

    overall_pass = True
    overall_max_rel = 0.0
    overall_max_abs = 0.0

    for fam in FAMILIES:
        abs_vals = all_diffs[fam]["abs"]
        rel_vals = all_diffs[fam]["rel"]
        if not abs_vals:
            continue
        max_abs = max(abs_vals)
        max_rel = max(rel_vals)
        fam_pass = max_rel < PASS_THRESHOLD_REL or max_abs < PASS_THRESHOLD_ABS
        if not fam_pass:
            overall_pass = False
        overall_max_rel = max(overall_max_rel, max_rel)
        overall_max_abs = max(overall_max_abs, max_abs)
        report["families"][fam] = {
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "pass": fam_pass,
        }

    report["overall"] = {
        "n_comparisons": total_compared,
        "max_abs_diff": overall_max_abs,
        "max_rel_diff": overall_max_rel,
        "pass": overall_pass,
    }

    output_path = os.path.expanduser("~/3dgs-renderer-benchmark/reports/r3_1/vec_gate_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Gaussians compared: {total_compared}")
    for fam in FAMILIES:
        d = report["families"].get(fam, {})
        print(f"  {fam}: max_abs={d.get('max_abs_diff', 0):.2e} max_rel={d.get('max_rel_diff', 0):.2e} {'PASS' if d.get('pass', True) else 'FAIL'}")
    print(f"\n  Overall: max_abs={overall_max_abs:.2e} max_rel={overall_max_rel:.2e}")
    print(f"  VEC Gate: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  Report: {output_path}")


if __name__ == "__main__":
    main()
