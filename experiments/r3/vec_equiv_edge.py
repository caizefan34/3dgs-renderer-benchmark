#!/usr/bin/env python3
"""
R3-VEC Equivalence Gate — enhanced edge-case comparison
========================================================
"""
import sys, os, math
import numpy as np
import torch

SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT)

import r3_certificate_runner as new

from r3_certificate_runner import _compute_faithful_work_weights_tensor

torch.manual_seed(42)


def _compute_work_weights_reference(g_indices, means2d_0, conics_0, opacities,
                                     tile_x, tile_y, tile_size, H, W):
    """Scalar reference: per-pixel forward replay with early termination."""
    # NOTE: This must exactly mirror the ORIGINAL scalar implementation.
    # We re-implement it from the git HEAD version.  For now, this test
    # focuses on the integer-count invariants which both must satisfy.
    if len(g_indices) == 0:
        return {}, {}
    G = len(g_indices)
    px_min = tile_x
    px_max = min(tile_x + tile_size, W)
    py_min = tile_y
    py_max = min(tile_y + tile_size, H)
    if px_max <= px_min or py_max <= py_min:
        return {}, {}

    w_color = {}
    w_unclamped = {}
    for px in range(px_min, px_max):
        for py in range(py_min, py_max):
            T = 1.0
            for pos in range(G):
                gi = int(g_indices[pos])
                dx = (px + 0.5) - float(means2d_0[gi, 0])
                dy = (py + 0.5) - float(means2d_0[gi, 1])
                xx, xy, yy = (float(conics_0[gi, 0]),
                              float(conics_0[gi, 1]),
                              float(conics_0[gi, 2]))
                sigma = 0.5 * (xx * dx * dx + yy * dy * dy) + xy * dx * dy
                a_raw = float(opacities[gi]) * math.exp(-max(sigma, -700.0))
                a = min(0.999, a_raw)
                if sigma < 0.0 or a < (1.0 / 255.0):
                    continue
                w_color[gi] = w_color.get(gi, 0) + 1
                if a_raw <= 0.999:
                    w_unclamped[gi] = w_unclamped.get(gi, 0) + 1
                T *= (1.0 - a)
                if T <= 1e-4:
                    break
    return w_color, w_unclamped


def _compute_faithful_work_weights_wrapper(gi_vals, means2d, conics, opacities,
                                            tile_idx, tile_size, H, W):
    """Call the tensor version exactly as _accumulate_tile_bounds does."""
    device = torch.device("cuda")
    tile_w = (W + tile_size - 1) // tile_size
    tile_x = (tile_idx % tile_w) * tile_size
    tile_y = (tile_idx // tile_w) * tile_size

    g_t = torch.from_numpy(gi_vals.astype(np.int64)).to(device)
    m_t = torch.from_numpy(means2d).to(device)
    c_t = torch.from_numpy(conics).to(device)
    o_t = torch.from_numpy(opacities).to(device)
    return _compute_faithful_work_weights_tensor(
        g_t, m_t, c_t, o_t, tile_x, tile_y, tile_size, H, W)


def main():
    print("=" * 72)
    print("R3-VEC Equivalence Gate — Edge-Case W-Equivalence")
    print("=" * 72)

    device = torch.device("cuda")
    N = 64
    H, W = 64, 64
    tile_size = 16
    tile_w = (W + tile_size - 1) // tile_size
    tile_h = (H + tile_size - 1) // tile_size
    n_tiles = tile_h * tile_w

    total_checks = 0
    failures = []

    for seed in range(100):
        rng = np.random.RandomState(seed)
        # Build a scene
        means2d = rng.rand(N, 2).astype(np.float32) * 64.0
        conics = (rng.rand(N, 3).astype(np.float32) * 0.8 + 0.2)
        opacities = rng.rand(N).astype(np.float32) * 0.99 + 0.01

        # Pick up to 12 random tiles (mix of border, center, etc.)
        chosen_tiles = rng.choice(n_tiles, size=min(n_tiles, 8), replace=False)
        for tile_idx in chosen_tiles:
            tx = (tile_idx % tile_w) * tile_size
            ty = (tile_idx // tile_w) * tile_size
            # Determine which gaussians intersect this tile (simple distance)
            # Use the same gaussian set for both implementations
            g_indices = np.arange(N)
            # Actually use a subset to keep it fast but include duplicates:
            # pick 3-8 gaussians sometimes with repeats
            k = int(rng.randint(3, 8))
            gi_vals = rng.choice(N, size=k, replace=False)
            # occasionally add duplicates
            if seed % 3 == 0 and k > 1:
                gi_vals = np.concatenate([gi_vals, gi_vals[:2]])

            q_t_val = rng.uniform(0.05, 1.0)

            # ---- Tensor implementation ----
            wc_t, wu_t = _compute_faithful_work_weights_wrapper(
                gi_vals, means2d, conics, opacities, tile_idx, tile_size, H, W)

            # ---- Scalar reference ----
            wc_r, wu_r = _compute_work_weights_reference(
                gi_vals, means2d, conics, opacities,
                tx, ty, tile_size, H, W)

            # Compare (aggregating tensor per-position to per-unique-GI)
            for i, gi in enumerate(gi_vals):
                total_checks += 1
                vt = int(wc_t[i])
                vr = int(wc_r.get(int(gi), 0))
                ut = int(wu_t[i])
                ur = int(wu_r.get(int(gi), 0))
                # The tensor returns per-depth-position values.
                # The reference returns per-GI totals (dict merges).
                # For duplicate GIs, compare AGGREGATED scatter sum.
                # For unique GIs, compare per-position value.
                # We detect duplicates:
                gi_count = np.sum(gi_vals == int(gi))
                if gi_count > 1:
                    # Skip per-position comparison for duplicates;
                    # we'll do aggregated comparison below.
                    total_checks -= 1
                    continue
                if vt != vr:
                    failures.append(f"seed={seed} tile={tile_idx} gi={gi}: "
                                    f"color tensor={vt} ref={vr}")
                if ut != ur:
                    failures.append(f"seed={seed} tile={tile_idx} gi={gi}: "
                                    f"unclamped tensor={ut} ref={ur}")

            # Aggregated duplicate check: scatter_add tensor per-GI and compare to ref
            u, inv = torch.unique(
                torch.from_numpy(gi_vals.astype(np.int64)).to(device),
                return_inverse=True
            )
            sum_wc = torch.zeros(len(u), dtype=torch.int32, device=device)
            sum_wu = torch.zeros(len(u), dtype=torch.int32, device=device)
            sum_wc.scatter_add_(0, inv, wc_t)
            sum_wu.scatter_add_(0, inv, wu_t)
            for j in range(len(u)):
                gi_u = int(u[j])
                vt_agg = int(sum_wc[j])
                vr = int(wc_r.get(gi_u, 0))
                ut_agg = int(sum_wu[j])
                ur = int(wu_r.get(gi_u, 0))
                total_checks += 1
                if vt_agg != vr:
                    failures.append(f"seed={seed} tile={tile_idx} gi={gi_u}: "
                                    f"color AGG tensor={vt_agg} ref={vr}")
                if ut_agg != ur:
                    failures.append(f"seed={seed} tile={tile_idx} gi={gi_u}: "
                                    f"unclamped AGG tensor={ut_agg} ref={ur}")

    print(f"  total_checks      = {total_checks}")
    print(f"  color_mismatch    = {sum(1 for f in failures if 'color' in f)}")
    print(f"  unclamped_mismatch= {sum(1 for f in failures if 'unclamped' in f)}")

    if failures:
        print("  W-EQUIVALENCE = FAIL")
        for f in failures[:10]:
            print(f"    {f}")
    else:
        print("  W-EQUIVALENCE = PASS")


if __name__ == "__main__":
    main()
