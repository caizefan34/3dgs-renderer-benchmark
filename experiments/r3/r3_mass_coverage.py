#!/usr/bin/env python3
"""
R3 Mass Coverage and Error Budgets (Corrected)

Section 15 (bound-based ranking) and Section 16 (certified error budgets)
computed from per-iteration gradient data stored in runner output.

These are post-hoc analyses of the per-Gaussian aggregate arrays: B_i and ||g_i||.
The tile-Gaussian error-budget argument (||sum g_{it}|| <= sum B_{it}) uses
triangle inequality and does NOT require per-tile empirical g_{it} values.

Storage note: per-Gaussian arrays (N ~ 900K) are stored as compressed npz files,
not JSON. This script reads them and produces aggregate tables.
"""

import os, sys, json, argparse, math, glob
import numpy as np


def compute_mass_coverage(actual_grads, bounds, quantiles=[0.20, 0.32, 0.50, 0.75]):
    """
    Section 15: Bound-based ranking (C2: uses per-Gaussian aggregate).
    
    Sort visible Gaussians by B_i ascending.
    Report actual gradient mass retained at each quantile.
    """
    if len(bounds) == 0 or len(actual_grads) == 0:
        return {"error": "No data"}
    
    sort_idx = np.argsort(bounds)
    sorted_bounds = bounds[sort_idx]
    sorted_actual = actual_grads[sort_idx]
    total_actual = float(sorted_actual.sum())
    
    if total_actual == 0:
        return {"error": "Zero total actual gradient"}
    
    cum_mass = np.cumsum(sorted_actual) / total_actual
    
    results = {}
    for q in quantiles:
        n = max(1, int(q * len(sorted_bounds)))
        retained = float(cum_mass[min(n, len(cum_mass)) - 1])
        results[f"retained_at_{int(q*100)}pct"] = retained
    
    # Retained mass curves for report
    for target in [0.95, 0.99, 0.999]:
        frac = int(np.searchsorted(cum_mass, target)) / max(len(cum_mass), 1)
        results[f"min_fraction_for_{int(target*100)}pct_mass"] = float(frac)
    
    return results


def compute_error_budgets(actual_grads, bounds, budgets=[0.001, 0.005, 0.01, 0.02, 0.05]):
    """
    Section 16: Certified error budgets (C2: uses triangle inequality).
    
    Sort by B_i ascending. For each budget epsilon:
        sum(skipped B_i) <= epsilon * sum(all B_i)
    
    Reports:
    - Gaussian fraction skippable under the budget
    - Actual gradient mass that would be omitted
    - Oracle comparison (what if we had perfect information?)
    """
    if len(bounds) == 0 or len(actual_grads) == 0:
        return {"error": "No data"}
    
    sort_idx = np.argsort(bounds)
    sorted_bounds = bounds[sort_idx]
    sorted_actual = actual_grads[sort_idx]
    
    total_bound = float(sorted_bounds.sum())
    total_actual = float(sorted_actual.sum())
    
    results = {}
    cum_bound = np.cumsum(sorted_bounds)
    
    for eps in budgets:
        threshold = eps * total_bound
        k = int(np.searchsorted(cum_bound, threshold, side='right'))
        
        gaussian_fraction = float(k / max(len(sorted_bounds), 1))
        actual_mass_omitted = float(sorted_actual[:k].sum() / total_actual) if total_actual > 0 else 0
        bound_budget_used = float(cum_bound[k-1] / total_bound) if k > 0 else 0
        
        results[f"eps_{eps*100:.1f}pct"] = {
            "gaussian_fraction_skippable": gaussian_fraction,
            "actual_mass_omitted": actual_mass_omitted,
            "bound_budget_used": bound_budget_used,
            "k_skipped": int(k),
            "total_N": len(sorted_bounds),
            "note_by_c2": "Per-Gaussian B_i; tile-Gaussian cert uses triangle inequality",
        }
    
    return results


def process_window(win_dir):
    """Process one window directory, loading npz gradient data."""
    npz_path = os.path.join(win_dir, "gradient_arrays.npz")
    if not os.path.exists(npz_path):
        return None
    
    data = np.load(npz_path)
    mass_results = {}
    budget_results = {}
    
    families = ["color_coarse", "color_tight", "opacity", "opacity_tight",
                "mean2d", "conic", "mean2d_sigmamin", "conic_sigmamin"]
    
    for fam in families:
        B_key = f"B_{fam}"
        g_key = f"g_color"  # map to correct g_actual key
        
        # Determine ground-truth key
        if "color" in fam:
            g_key = "g_color"
        elif "opacity" in fam:
            g_key = "g_opacity"
        elif "mean2d" in fam:
            g_key = "g_mean2d"
        elif "conic" in fam:
            g_key = "g_conic"
        
        if B_key in data and g_key in data:
            B_arr = data[B_key]
            g_arr = data[g_key]
            
            # Filter visible (non-zero) Gaussians
            vis = g_arr > 0
            if vis.sum() > 0:
                mass_results[fam] = compute_mass_coverage(g_arr[vis], B_arr[vis])
                budget_results[fam] = compute_error_budgets(g_arr[vis], B_arr[vis])
    
    return {
        "mass_coverage": mass_results,
        "error_budgets": budget_results,
        "N_gaussians": int(data.get("N", 0)),
        "visible_gaussians": int((data.get("g_color", np.zeros(1)) > 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="R3 mass coverage and error budgets (corrected)"
    )
    parser.add_argument("--input", required=True, help="Top-level R3 results dir")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Find window subdirectories
    windows = {}
    for name in sorted(os.listdir(args.input)):
        sub = os.path.join(args.input, name)
        if os.path.isdir(sub):
            windows[name] = sub
    
    mass_aggregate = {}
    budget_aggregate = {}
    
    for win_name, win_dir in windows.items():
        result = process_window(win_dir)
        if result:
            mass_aggregate[win_name] = result["mass_coverage"]
            budget_aggregate[win_name] = result["error_budgets"]
            print(f"  {win_name}: {result.get('visible_gaussians', '?')} visible "
                  f"Gaussians, {len(result.get('mass_coverage', {}))} families")
    
    # Save
    def save_json(name, data):
        path = os.path.join(args.output, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved: {path}")
    
    save_json("gradient_mass_coverage.json", {
        "mass_coverage": mass_aggregate,
        "note": "Per-Gaussian aggregate B_i and ||g_i||. "
                "Tile-Gaussian error budget uses triangle inequality.",
    })
    save_json("certified_error_budgets.json", {
        "error_budgets": budget_aggregate,
    })
    save_json("tile_gaussian_certificate.json", {
        "note": "Tile-Gaussian level data computed in runner (exact-zero counts). "
                "Per-tile B_it values available in forward pass only.",
        "per_window_data": {
            win: {"exact_zero_fraction": None} for win in windows
        },
    })


if __name__ == "__main__":
    main()
