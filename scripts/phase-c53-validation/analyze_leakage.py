#!/usr/bin/env python3
"""
Phase C53-Validation — Leakage Analysis

Central experiment: Does visibility predict future actual work BEYOND future visibility?

Tests:
1. current visibility → future actual work (tile_work)
2. future visibility → future actual work (contemporaneous)
3. current visibility → RESIDUAL future work (after regressing out future visibility)

Also:
4. Time-shifted comparison: visibility(t) → work(t+Δ) vs visibility(t+Δ) → work(t+Δ)
"""
import json, sys
from pathlib import Path
import numpy as np

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery")
OUT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
DELTAS = [10, 50, 100]


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def linear_regression_residual(x, y):
    """Regress y on x, return residuals. Uses least squares."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        return y - y_pred
    except Exception:
        return y - y.mean()


def analyze_scene(scene, suffix=""):
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        return None

    data = np.load(filepath, allow_pickle=True)
    results = {"scene": scene + suffix, "checkpoints": {}}

    for cp in CHECKPOINTS:
        prefix = f"cp{cp}"
        if f"{prefix}_ids" not in data:
            continue

        # Get current signals
        vis_current = data[f"{prefix}_s_visibility_count"].astype(np.float64)
        rad_current = data[f"{prefix}_s_screen_radius_mean"].astype(np.float64)
        ema_grad = data[f"{prefix}_s_ema_grad_norm"].astype(np.float64)
        opacity = data[f"{prefix}_s_opacity"].astype(np.float64)
        scale = data[f"{prefix}_s_scale_norm"].astype(np.float64)

        cp_results = {}
        for d in DELTAS:
            d_prefix = f"{prefix}_d{d}"
            if f"{d_prefix}_alive" not in data:
                continue
            alive = data[f"{d_prefix}_alive"]

            # Future targets
            future_vis = data[f"{d_prefix}_vis_count"].astype(np.float64)
            future_tiles = data[f"{d_prefix}_tiles_mean"].astype(np.float64)
            future_area = data[f"{d_prefix}_area_mean"].astype(np.float64)
            future_update = data[f"{d_prefix}_update_xyz"].astype(np.float64)
            future_grad = data[f"{d_prefix}_grad_sum"].astype(np.float64)

            mask = alive.copy()

            d_results = {}

            # Test 1: current visibility → future actual work
            d_results["test1_current_vis_to_future_work"] = {
                "tile_work": safe_pearson(vis_current[mask], future_tiles[mask]),
                "pixel_work": safe_pearson(vis_current[mask], future_area[mask]),
                "update": safe_pearson(vis_current[mask], future_update[mask]),
            }

            # Test 2: future visibility → future actual work (contemporaneous upper bound)
            d_results["test2_future_vis_to_future_work"] = {
                "tile_work": safe_pearson(future_vis[mask], future_tiles[mask]),
                "pixel_work": safe_pearson(future_vis[mask], future_area[mask]),
                "update": safe_pearson(future_vis[mask], future_update[mask]),
            }

            # Test 3: current visibility → RESIDUAL future work
            # First regress future_work on future_vis, get residuals
            for work_name, work_vals in [("tile_work", future_tiles), ("pixel_work", future_area), ("update", future_update)]:
                w = work_vals[mask]
                fv = future_vis[mask]
                residual = linear_regression_residual(fv, w)
                d_results[f"test3_current_vis_to_residual_{work_name}"] = {
                    "pearson": safe_pearson(vis_current[mask], residual),
                    "interpretation": "If meaningful, vis contains info about future work beyond future visibility"
                }

            # Test 4: Time-shifted comparison
            d_results["test4_time_shifted"] = {
                "predictive_vis_to_tile": safe_pearson(vis_current[mask], future_tiles[mask]),
                "contemporaneous_vis_to_tile": safe_pearson(future_vis[mask], future_tiles[mask]),
                "gap": safe_pearson(future_vis[mask], future_tiles[mask]) - safe_pearson(vis_current[mask], future_tiles[mask]),
            }

            # Test 5: Other signals to residual tile work
            fv = future_vis[mask]
            ft = future_tiles[mask]
            residual_tiles = linear_regression_residual(fv, ft)
            d_results["test5_other_signals_to_residual_tile"] = {
                "screen_radius_mean": safe_pearson(rad_current[mask], residual_tiles),
                "ema_grad_norm": safe_pearson(ema_grad[mask], residual_tiles),
                "opacity": safe_pearson(opacity[mask], residual_tiles),
                "scale_norm": safe_pearson(scale[mask], residual_tiles),
            }

            # Test 6: Current visibility → future vis (the original C53-Discovery result)
            d_results["test6_current_vis_to_future_vis"] = {
                "pearson": safe_pearson(vis_current[mask], future_vis[mask]),
            }

            cp_results[str(d)] = d_results
        results["checkpoints"][str(cp)] = cp_results

    return results


def main():
    print("=" * 110)
    print("Phase C53-Validation — Leakage Analysis")
    print("Does visibility predict future ACTUAL WORK beyond future visibility?")
    print("=" * 110)

    scenes = ["room", "garden", "bicycle", "room_seed123", "garden_seed123"]
    all_results = {}

    for scene in scenes:
        print(f"\n--- {scene} ---")
        parts = scene.split("_seed")
        base_scene = parts[0]
        suffix = f"_seed{parts[1]}" if len(parts) > 1 else ""
        results = analyze_scene(base_scene, suffix)
        if results:
            all_results[scene] = results

            # Print key results for delta=50
            for d in DELTAS:
                d_str = str(d)
                cp_means = {k: [] for k in [
                    "test1_tile", "test1_pixel", "test1_update",
                    "test2_tile", "test2_pixel",
                    "test3_tile", "test3_pixel", "test3_update",
                    "test4_pred", "test4_contemp", "test4_gap",
                    "test6",
                ]}
                for cp in CHECKPOINTS:
                    cp_str = str(cp)
                    if cp_str in results.get("checkpoints", {}):
                        cp_data = results["checkpoints"][cp_str]
                        if d_str in cp_data:
                            dd = cp_data[d_str]
                            cp_means["test1_tile"].append(dd["test1_current_vis_to_future_work"]["tile_work"])
                            cp_means["test1_pixel"].append(dd["test1_current_vis_to_future_work"]["pixel_work"])
                            cp_means["test1_update"].append(dd["test1_current_vis_to_future_work"]["update"])
                            cp_means["test2_tile"].append(dd["test2_future_vis_to_future_work"]["tile_work"])
                            cp_means["test2_pixel"].append(dd["test2_future_vis_to_future_work"]["pixel_work"])
                            cp_means["test3_tile"].append(dd["test3_current_vis_to_residual_tile_work"]["pearson"])
                            cp_means["test3_pixel"].append(dd["test3_current_vis_to_residual_pixel_work"]["pearson"])
                            cp_means["test3_update"].append(dd["test3_current_vis_to_residual_update"]["pearson"])
                            cp_means["test4_pred"].append(dd["test4_time_shifted"]["predictive_vis_to_tile"])
                            cp_means["test4_contemp"].append(dd["test4_time_shifted"]["contemporaneous_vis_to_tile"])
                            cp_means["test4_gap"].append(dd["test4_time_shifted"]["gap"])
                            cp_means["test6"].append(dd["test6_current_vis_to_future_vis"]["pearson"])

                if cp_means["test1_tile"]:
                    print(f"\n  Δ={d} (mean across {len(cp_means['test1_tile'])} checkpoints):")
                    print(f"    Test 1: current_vis → future_work:  tile={np.mean(cp_means['test1_tile']):.3f}, pixel={np.mean(cp_means['test1_pixel']):.3f}, update={np.mean(cp_means['test1_update']):.3f}")
                    print(f"    Test 2: future_vis → future_work:    tile={np.mean(cp_means['test2_tile']):.3f}, pixel={np.mean(cp_means['test2_pixel']):.3f}")
                    print(f"    Test 3: current_vis → RESIDUAL:      tile={np.mean(cp_means['test3_tile']):.3f}, pixel={np.mean(cp_means['test3_pixel']):.3f}, update={np.mean(cp_means['test3_update']):.3f}")
                    print(f"    Test 4: pred={np.mean(cp_means['test4_pred']):.3f}, contemp={np.mean(cp_means['test4_contemp']):.3f}, gap={np.mean(cp_means['test4_gap']):.3f}")
                    print(f"    Test 6: current_vis → future_vis:    {np.mean(cp_means['test6']):.3f} (ORIGINAL C53 RESULT)")

    # Cross-scene summary
    print("\n\n" + "=" * 110)
    print("CROSS-SCENE LEAKAGE SUMMARY (mean across room, garden, bicycle, Δ=50)")
    print("=" * 110)
    primary = ["room", "garden", "bicycle"]
    for test_name, label in [
        ("test1_tile", "current_vis → future_tile_work"),
        ("test2_tile", "future_vis → future_tile_work"),
        ("test3_tile", "current_vis → RESIDUAL tile_work"),
        ("test1_update", "current_vis → future_update"),
        ("test3_update", "current_vis → RESIDUAL update"),
        ("test6", "current_vis → future_vis (C53 ORIGINAL)"),
    ]:
        vals = []
        for scene in primary:
            if scene in all_results:
                for cp in CHECKPOINTS:
                    cp_str = str(cp)
                    if cp_str in all_results[scene].get("checkpoints", {}):
                        cp_data = all_results[scene]["checkpoints"][cp_str]
                        if "50" in cp_data:
                            dd = cp_data["50"]
                            if test_name == "test1_tile":
                                vals.append(dd["test1_current_vis_to_future_work"]["tile_work"])
                            elif test_name == "test2_tile":
                                vals.append(dd["test2_future_vis_to_future_work"]["tile_work"])
                            elif test_name == "test3_tile":
                                vals.append(dd["test3_current_vis_to_residual_tile_work"]["pearson"])
                            elif test_name == "test1_update":
                                vals.append(dd["test1_current_vis_to_future_work"]["update"])
                            elif test_name == "test3_update":
                                vals.append(dd["test3_current_vis_to_residual_update"]["pearson"])
                            elif test_name == "test6":
                                vals.append(dd["test6_current_vis_to_future_vis"]["pearson"])
        if vals:
            print(f"  {label:<45} mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, n={len(vals)}")

    # Save
    out_file = OUT_DIR / "leakage_analysis.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
