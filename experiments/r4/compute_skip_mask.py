#!/usr/bin/env python3
"""
R4 compute_skip_mask: Python-side certificate skip mask computation.

Given the forward pass outputs (means2d, conics, colors, opacities,
tile_offsets, flatten_ids, render_alphas), compute a boolean skip_mask
tensor [n_isects] that marks which (tile, Gaussian) intersections can
be safely skipped in the backward pass.

Uses the R3.1 corrected certificate formulas:
  - c_norm = ||colors[g]|| (TRUE SH color norm, not conic proxy)
  - C_max_t = max color norm in tile
  - 4 family bounds: color, opacity, mean2d (sigmamin), conic (sigmamin)
  - Greedy skip selection: rank by max normalized contribution, ascending
  - Budget: B^fam_it <= epsilon * total_B^fam for ALL 4 families

The skip_mask is indexed by intersection position in flatten_ids,
matching the CUDA kernel's indexing (isect_idx = batch_end - t).
"""

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple


def compute_per_intersection_bounds(
    means2d: torch.Tensor,       # [C, N, 2]
    conics: torch.Tensor,        # [C, N, 3]
    colors: torch.Tensor,        # [C, N, 3] or [C, N, K, 3]
    opacities: torch.Tensor,     # [C, N] or [N]
    tile_offsets: torch.Tensor,  # [C, tile_h, tile_w]
    flatten_ids: torch.Tensor,   # [n_isects]
    tile_size: int,
    image_width: int,
    image_height: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-intersection bounds for all 4 derivative families.

    Returns dict with:
      - B_color: [n_isects] color bound
      - B_opacity: [n_isects] opacity bound
      - B_mean2d: [n_isects] mean2d bound
      - B_conic: [n_isects] conic bound
      - W_isect: [n_isects] approximate pixel-lane count (work weight)
      - tile_ids: [n_isects] tile index per intersection
    """
    device = means2d.device
    C, N = means2d.shape[0], means2d.shape[1]
    n_isects = flatten_ids.shape[0]

    tile_h = tile_offsets.shape[1]
    tile_w = tile_offsets.shape[2]

    # Get color norm — handle both [C,N,3] and [C,N,K,3] SH formats
    if colors.dim() == 3:
        c_norm_all = colors[0].norm(dim=-1)  # [N]
    elif colors.dim() == 4:
        c_norm_all = colors[0, :, 0].norm(dim=-1)  # [N] - use DC component
    else:
        c_norm_all = torch.ones(N, device=device)

    # Conic sigma_min = min eigenvalue of conic matrix
    # conic = [a, b, c], matrix = [[a, b], [b, c]]
    # eigenvalues = (a+c ± sqrt((a-c)^2 + 4b^2)) / 2
    a = conics[0, :, 0]  # [N]
    b = conics[0, :, 1]  # [N]
    c = conics[0, :, 2]  # [N]
    sigma_min_all = (a + c - torch.sqrt((a - c) ** 2 + 4 * b ** 2)) / 2
    sigma_min_all = sigma_min_all.clamp_min(1e-8)  # avoid div-by-zero

    # Opacities
    if opacities.dim() == 1:
        opacity_all = opacities  # [N]
    else:
        opacity_all = opacities[0]  # [N]

    # Per-intersection quantities
    g_ids = flatten_ids  # [n_isects]
    
    # For very large n_isects, compute on CPU to avoid GPU OOM
    if n_isects > 200_000_000:
        # Move per-Gaussian quantities to CPU (they're small: [N])
        c_norm_all_cpu = c_norm_all.cpu()
        sigma_min_all_cpu = sigma_min_all.cpu()
        opacity_all_cpu = opacity_all.cpu()
        g_ids_cpu = g_ids.cpu()
        
        c_norm = c_norm_all_cpu[g_ids_cpu]
        sigma_min = sigma_min_all_cpu[g_ids_cpu]
        opacity = opacity_all_cpu[g_ids_cpu]
        
        # tile_ids on CPU
        tile_offsets_flat_cpu = tile_offsets[0].flatten().cpu()
        tile_ids = torch.searchsorted(tile_offsets_flat_cpu, torch.arange(n_isects), right=True) - 1
        tile_ids = tile_ids.clamp_min(0)
        
        # C_max per tile on CPU
        C_max_per_tile = torch.zeros(tile_h * tile_w)
        C_max_per_tile.scatter_reduce_(0, tile_ids, c_norm, reduce="amax", include_self=True)
        C_max_t = C_max_per_tile[tile_ids]
        
        # All bounds on CPU
        alpha = opacity
        T = 1.0
        B_color = c_norm * alpha * T
        B_opacity = (c_norm + C_max_t) * T * alpha
        B_mean2d = c_norm * alpha * T / sigma_min
        pixel_dist_sq = float(tile_size) ** 2 / 4.0
        B_conic = c_norm * alpha * T * pixel_dist_sq / sigma_min
        W_isect = torch.ones(n_isects) * float(tile_size) ** 2
        
        return {
            "B_color": B_color, "B_opacity": B_opacity,
            "B_mean2d": B_mean2d, "B_conic": B_conic,
            "W_isect": W_isect, "tile_ids": tile_ids,
            "c_norm": c_norm, "sigma_min": sigma_min,
            "C_max_t": C_max_t, "alpha": alpha,
        }
    
    # Normal GPU path for smaller n_isects
    c_norm = c_norm_all[g_ids]        # [n_isects]
    sigma_min = sigma_min_all[g_ids]   # [n_isects]
    opacity = opacity_all[g_ids]       # [n_isects]

    # Compute per-tile C_max_t
    # Build tile_ids: for each intersection, which tile does it belong to?
    # Vectorized: use searchsorted on the flattened tile_offsets
    tile_offsets_flat = tile_offsets[0].flatten()  # [tile_h * tile_w]
    # tile_offsets_flat[i] = start index in flatten_ids for tile i
    # For intersection at position j, tile_id = i where tile_offsets_flat[i] <= j < tile_offsets_flat[i+1]
    # Use searchsorted to find the tile_id for each intersection
    tile_ids = torch.searchsorted(tile_offsets_flat, torch.arange(n_isects, device=device), right=True) - 1
    tile_ids = tile_ids.clamp_min(0)

    # C_max_t = max color norm per tile (via scatter_reduce)
    C_max_per_tile = torch.zeros(tile_h * tile_w, device=device)
    C_max_per_tile.scatter_reduce_(0, tile_ids, c_norm, reduce="amax", include_self=True)
    C_max_t = C_max_per_tile[tile_ids]  # [n_isects]

    # Approximate alpha: opacity * gaussian_overlap_factor
    # We use a conservative alpha = opacity (max possible alpha)
    alpha = opacity  # [n_isects] — conservative upper bound

    # Approximate transmittance T_i before Gaussian i
    # Conservative: use T=1.0 (worst case, all contributions at max)
    # This makes bounds larger → fewer skips → conservative/safe
    T = 1.0

    # Bound formulas (conservative, from R3.1 certificate):
    # B_color = ||c_i|| * alpha_i * T_i
    B_color = c_norm * alpha * T

    # B_opacity = (||c_i|| + C_max_t) * T_i * alpha_i
    B_opacity = (c_norm + C_max_t) * T * alpha

    # B_mean2d = ||c_i|| * alpha_i * T_i / sigma_min
    B_mean2d = c_norm * alpha * T / sigma_min

    # B_conic = ||c_i|| * alpha_i * T_i * pixel_dist^2 / sigma_min
    # pixel_dist ≈ tile_size / 2 (half-tile radius)
    pixel_dist_sq = float(tile_size) ** 2 / 4.0
    B_conic = c_norm * alpha * T * pixel_dist_sq / sigma_min

    # Work weight W_isect: approximate pixel-lane count
    # For a Gaussian in a tile, the number of active pixel lanes is
    # roughly proportional to its 2D footprint area / pixel area
    # Use sigma_min as a proxy: more spread = more lanes
    # This is approximate; the real count comes from the forward pass
    W_isect = torch.ones(n_isects, device=device) * float(tile_size) ** 2

    return {
        "B_color": B_color,
        "B_opacity": B_opacity,
        "B_mean2d": B_mean2d,
        "B_conic": B_conic,
        "W_isect": W_isect,
        "tile_ids": tile_ids,
        "c_norm": c_norm,
        "sigma_min": sigma_min,
        "C_max_t": C_max_t,
        "alpha": alpha,
    }


def compute_skip_mask(
    means2d: torch.Tensor,       # [C, N, 2]
    conics: torch.Tensor,        # [C, N, 3]
    colors: torch.Tensor,        # [C, N, 3]
    opacities: torch.Tensor,     # [C, N]
    tile_offsets: torch.Tensor,  # [C, tile_h, tile_w]
    flatten_ids: torch.Tensor,   # [n_isects]
    tile_size: int,
    image_width: int,
    image_height: int,
    budget: float = 0.05,
) -> torch.Tensor:
    """
    Compute certificate-based skip mask for (tile, Gaussian) intersections.

    Returns: bool tensor [n_isects] — True means safe to skip gradient work.

    Algorithm (R3.1 joint skip):
    1. Compute per-intersection bounds for 4 families
    2. Compute total bounds per family
    3. Rank intersections by max normalized contribution (ascending)
    4. Greedily add to skip set if ALL families stay within budget
    """
    # Clear CUDA cache to free memory from forward pass
    torch.cuda.empty_cache()

    bounds = compute_per_intersection_bounds(
        means2d, conics, colors, opacities,
        tile_offsets, flatten_ids,
        tile_size, image_width, image_height,
    )

    n_isects = flatten_ids.shape[0]
    device = means2d.device

    # For very large n_isects (>200M), move bounds to CPU to avoid OOM
    use_cpu = n_isects > 200_000_000
    if use_cpu:
        print(f"  [skip_mask] n_isects={n_isects} > 200M, using CPU for bounds processing")

    B_color = bounds["B_color"]
    B_opacity = bounds["B_opacity"]
    B_mean2d = bounds["B_mean2d"]
    B_conic = bounds["B_conic"]
    W_isect = bounds["W_isect"]

    if use_cpu:
        B_color = B_color.cpu()
        B_opacity = B_opacity.cpu()
        B_mean2d = B_mean2d.cpu()
        B_conic = B_conic.cpu()
        W_isect = W_isect.cpu()
        compute_device = torch.device("cpu")
    else:
        compute_device = device

    # Total bounds (denominator for epsilon)
    total_B_color = B_color.sum().item()
    total_B_opacity = B_opacity.sum().item()
    total_B_mean2d = B_mean2d.sum().item()
    total_B_conic = B_conic.sum().item()

    # Guard against zero totals
    tc = max(total_B_color, 1e-12)
    to = max(total_B_opacity, 1e-12)
    tm = max(total_B_mean2d, 1e-12)
    tk = max(total_B_conic, 1e-12)

    # Max normalized contribution per intersection
    max_norm = torch.stack([
        B_color / tc,
        B_opacity / to,
        B_mean2d / tm,
        B_conic / tk,
    ], dim=-1).max(dim=-1).values  # [n_isects]

    # Sort ascending (smallest contribution first → most skippable)
    sorted_idx = torch.argsort(max_norm)

    # Vectorized greedy skip selection:
    # Sort bounds by the same order, compute cumulative sums,
    # then find the cutoff point where ALL family budgets are exceeded.
    B_color_sorted = B_color[sorted_idx]
    B_opacity_sorted = B_opacity[sorted_idx]
    B_mean2d_sorted = B_mean2d[sorted_idx]
    B_conic_sorted = B_conic[sorted_idx]

    # Cumulative sums (prefix sums) on GPU
    cum_color = torch.cumsum(B_color_sorted, dim=0)
    cum_opacity = torch.cumsum(B_opacity_sorted, dim=0)
    cum_mean2d = torch.cumsum(B_mean2d_sorted, dim=0)
    cum_conic = torch.cumsum(B_conic_sorted, dim=0)

    # Budget thresholds
    eps = budget
    budget_color = eps * tc
    budget_opacity = eps * to
    budget_mean2d = eps * tm
    budget_conic = eps * tk

    # Find the LAST index where ALL cumulative sums are within budget
    # i.e., the largest k such that cum_color[k] <= budget_color AND
    #        cum_opacity[k] <= budget_opacity AND ...
    within_budget = (
        (cum_color <= budget_color) &
        (cum_opacity <= budget_opacity) &
        (cum_mean2d <= budget_mean2d) &
        (cum_conic <= budget_conic)
    )

    # The skip set is all indices [0, k] where k is the last True in within_budget
    # Since the greedy algorithm stops at the first violation, we take
    # the contiguous prefix [0, k] of sorted indices
    # Find k: the last True index
    if within_budget.any():
        # Find the first False after the initial True run
        first_violation = (~within_budget).nonzero(as_tuple=True)[0]
        if len(first_violation) > 0:
            k = first_violation[0].item()  # exclusive end
        else:
            k = n_isects  # all within budget
    else:
        k = 0  # nothing can be skipped

    # Mark sorted indices [0, k) as skipped
    skip_mask = torch.zeros(n_isects, dtype=torch.bool, device=compute_device)
    if k > 0:
        skip_mask[sorted_idx[:k]] = True

    skip_count = k
    skip_weighted = W_isect[sorted_idx[:k]].sum().item()

    # Move back to original device if we used CPU
    if use_cpu:
        skip_mask = skip_mask.to(device)

    total_weighted = W_isect.sum().item()
    skip_frac = skip_count / n_isects if n_isects > 0 else 0.0
    skip_weighted_frac = skip_weighted / total_weighted if total_weighted > 0 else 0.0

    print(f"  [skip_mask] budget={budget:.1%} | "
          f"skipped {skip_count}/{n_isects} pairs ({skip_frac:.2%}), "
          f"weighted work {skip_weighted_frac:.2%}")

    return skip_mask


if __name__ == "__main__":
    # Quick self-test with synthetic data
    import sys

    print("=== compute_skip_mask self-test ===")
    N = 100
    C = 1
    tile_size = 16
    H, W = 64, 64
    tile_h, tile_w = H // tile_size, W // tile_size

    means2d = torch.randn(C, N, 2, device="cuda") * 32 + 32
    conics = torch.randn(C, N, 3, device="cuda").abs()  # positive conics
    conics[..., 1] = 0  # diagonal → sigma_min = min(a, c)
    colors = torch.randn(C, N, 3, device="cuda").abs()
    opacities = torch.rand(C, N, device="cuda")
    flatten_ids = torch.randint(0, N, (200,), device="cuda")
    tile_offsets = torch.zeros(C, tile_h, tile_w, dtype=torch.int32, device="cuda")

    # Fill tile_offsets with cumulative counts
    counts_per_tile = torch.randint(0, 5, (tile_h * tile_w,), device="cuda")
    cumsum = counts_per_tile.cumsum(0).to(torch.int32)
    for ti in range(tile_h * tile_w):
        th, tw = ti // tile_w, ti % tile_w
        tile_offsets[0, th, tw] = cumsum[ti - 1] if ti > 0 else 0
    n_isects = counts_per_tile.sum().item()
    flatten_ids = torch.randint(0, N, (n_isects,), device="cuda")

    for budget in [0.01, 0.05, 0.10]:
        mask = compute_skip_mask(
            means2d, conics, colors, opacities,
            tile_offsets, flatten_ids,
            tile_size, W, H, budget=budget,
        )
        print(f"  budget={budget}: skipped {mask.sum().item()}/{n_isects}")
    print("Self-test done.")
