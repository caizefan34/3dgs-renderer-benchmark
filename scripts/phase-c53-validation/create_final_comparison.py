#!/usr/bin/env python3
"""
Phase C53-Validation — Final Comparison & Cross-Scene Analysis

Produces the final cross-scene comparison and signal ranking.
"""
import json
from pathlib import Path
import numpy as np

OUT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS = ["prev_grad_norm", "ema_grad_norm", "visibility_count",
           "screen_radius_mean", "opacity", "scale_norm", "age"]


def load_json(name):
    p = OUT_DIR / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def main():
    print("=" * 110)
    print("Phase C53-Validation — Final Comparison & Signal Ranking")
    print("=" * 110)

    fw = load_json("future_work_analysis.json")
    leak = load_json("leakage_analysis.json")
    age = load_json("age_analysis.json")
    horizon = load_json("horizon_analysis.json")
    backward = load_json("backward_work_analysis.json")
    intersection = load_json("intersection_work_analysis.json")

    if not fw:
        print("ERROR: future_work_analysis.json not found. Run analyze_future_work.py first.")
        return

    summaries = fw.get("summaries", {})
    primary = ["room", "garden", "bicycle"]

    # === TABLE 1: Leakage-Free Prediction ===
    print("\n" + "=" * 110)
    print("TABLE 1 — Leakage-Free Prediction (mean Pearson, Δ=50, across room/garden/bicycle)")
    print("=" * 110)
    print(f"  {'Signal':<25} {'→ tile_work':>14} {'→ pixel_work':>14} {'→ update':>10} {'→ grad':>10} {'→ vis(LEAK)':>12}")
    table1 = {}
    for sig in SIGNALS:
        vals = {}
        for tname in ["future_tile_work", "future_pixel_work", "future_update", "future_grad", "future_vis"]:
            v = []
            for scene in primary:
                if scene in summaries and "50" in summaries[scene].get("deltas", {}):
                    d_data = summaries[scene]["deltas"]["50"]
                    if tname in d_data and sig in d_data[tname]:
                        v.append(d_data[tname][sig]["pearson"])
            vals[tname] = float(np.mean(v)) if v else 0
        table1[sig] = vals
        print(f"  {sig:<25} {vals['future_tile_work']:>14.3f} {vals['future_pixel_work']:>14.3f} "
              f"{vals['future_update']:>10.3f} {vals['future_grad']:>10.3f} {vals['future_vis']:>12.3f}")

    # === TABLE 2: Work Type ===
    print("\n" + "=" * 110)
    print("TABLE 2 — Work Type (mean Pearson, Δ=50, across room/garden/bicycle)")
    print("=" * 110)
    if intersection:
        print(f"  {'Signal':<25} {'Forward(tiles)':>16} {'Intersection':>14} {'Backward(update)':>18} {'Backward(grad)':>16}")
        table2 = {}
        for sig in SIGNALS:
            vals = {}
            for layer in ["intersection_tiles", "rasterization_area", "backward_update", "backward_grad"]:
                v = []
                for scene in primary:
                    if scene in intersection:
                        key = "delta_50"
                        if key in intersection[scene] and layer in intersection[scene][key] and sig in intersection[scene][key][layer]:
                            v.append(intersection[scene][key][layer][sig]["pearson"])
                vals[layer] = float(np.mean(v)) if v else 0
            table2[sig] = vals
            print(f"  {sig:<25} {vals['rasterization_area']:>16.3f} {vals['intersection_tiles']:>14.3f} "
                  f"{vals['backward_update']:>18.3f} {vals['backward_grad']:>16.3f}")

    # === TABLE 3: Horizon ===
    print("\n" + "=" * 110)
    print("TABLE 3 — Horizon (mean Pearson, target=tile_work, across room/garden/bicycle)")
    print("=" * 110)
    if horizon:
        print(f"  {'Signal':<25} {'Δ=10':>10} {'Δ=50':>10} {'Δ=100':>10} {'Trend':>10}")
        table3 = {}
        for sig in SIGNALS:
            vals = []
            for d in [10, 50, 100]:
                v = []
                for scene in primary:
                    if scene in horizon:
                        key = f"delta_{d}"
                        if key in horizon[scene] and "future_tile_work" in horizon[scene][key] and sig in horizon[scene][key]["future_tile_work"]:
                            v.append(horizon[scene][key]["future_tile_work"][sig])
                vals.append(float(np.mean(v)) if v else 0)
            trend = "↑" if vals[2] > vals[0] + 0.01 else "↓" if vals[2] < vals[0] - 0.01 else "→"
            table3[sig] = {"d10": vals[0], "d50": vals[1], "d100": vals[2], "trend": trend}
            print(f"  {sig:<25} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {trend:>10}")

    # === TABLE 4: Age ===
    print("\n" + "=" * 110)
    print("TABLE 4 — Age (best signal per age group, target=tile_work, Δ=50, across room/garden/bicycle)")
    print("=" * 110)
    age_labels = ["<100", "100-500", "500-2000", ">2000"]
    table4 = {}
    if age:
        for ag_label in age_labels:
            best_sig, best_val = "none", -999
            all_sig_vals = {}
            for sig in SIGNALS:
                v = []
                for scene in primary:
                    if scene in age and ag_label in age[scene].get("age_groups", {}):
                        ag_data = age[scene]["age_groups"][ag_label]
                        for cp in [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]:
                            cp_str = str(cp)
                            if cp_str in ag_data.get("checkpoints", {}):
                                cp_data = ag_data["checkpoints"][cp_str]
                                if "50" in cp_data and "future_tile_work" in cp_data["50"] and sig in cp_data["50"]["future_tile_work"]:
                                    v.append(cp_data["50"]["future_tile_work"][sig])
                m = float(np.mean(v)) if v else 0
                all_sig_vals[sig] = m
                if m > best_val:
                    best_val = m
                    best_sig = sig
            table4[ag_label] = {"best_signal": best_sig, "tile_work_corr": best_val, "all_signals": all_sig_vals}
            print(f"  Age {ag_label:<12}: best={best_sig:<25} Pearson={best_val:.3f}")

    # === TABLE 5: Cross-Scene ===
    print("\n" + "=" * 110)
    print("TABLE 5 — Cross-Scene (Pearson, target=tile_work, Δ=50)")
    print("=" * 110)
    print(f"  {'Signal':<25} {'Room':>8} {'Garden':>8} {'Bicycle':>8} {'Mean':>8} {'Std':>8}")
    table5 = {}
    for sig in SIGNALS:
        vals = {}
        for scene in primary:
            if scene in summaries and "50" in summaries[scene].get("deltas", {}):
                d_data = summaries[scene]["deltas"]["50"]
                if "future_tile_work" in d_data and sig in d_data["future_tile_work"]:
                    vals[scene] = d_data["future_tile_work"][sig]["pearson"]
                else:
                    vals[scene] = 0
            else:
                vals[scene] = 0
        mean_val = float(np.mean(list(vals.values())))
        std_val = float(np.std(list(vals.values())))
        table5[sig] = {**vals, "mean": mean_val, "std": std_val}
        print(f"  {sig:<25} {vals.get('room',0):>8.3f} {vals.get('garden',0):>8.3f} "
              f"{vals.get('bicycle',0):>8.3f} {mean_val:>8.3f} {std_val:>8.3f}")

    # === TABLE 6: Visibility Leakage Control ===
    print("\n" + "=" * 110)
    print("TABLE 6 — Visibility Leakage Control (mean across room/garden/bicycle, Δ=50)")
    print("=" * 110)
    table6 = {}
    if leak:
        test_names = [
            ("test1_tile", "current_vis → future_tile_work", "test1_current_vis_to_future_work", "tile_work"),
            ("test2_tile", "future_vis → future_tile_work", "test2_future_vis_to_future_work", "tile_work"),
            ("test3_tile", "current_vis → RESIDUAL tile_work", "test3_current_vis_to_residual_tile_work", "pearson"),
            ("test1_update", "current_vis → future_update", "test1_current_vis_to_future_work", "update"),
            ("test3_update", "current_vis → RESIDUAL update", "test3_current_vis_to_residual_update", "pearson"),
            ("test6", "current_vis → future_vis (C53 ORIGINAL)", "test6_current_vis_to_future_vis", "pearson"),
        ]
        for label, desc, test_key, field in test_names:
            v = []
            for scene in primary:
                if scene in leak:
                    for cp in [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]:
                        cp_str = str(cp)
                        if cp_str in leak[scene].get("checkpoints", {}):
                            cp_data = leak[scene]["checkpoints"][cp_str]
                            if "50" in cp_data and test_key in cp_data["50"]:
                                if field in cp_data["50"][test_key]:
                                    v.append(cp_data["50"][test_key][field])
            m = float(np.mean(v)) if v else 0
            s = float(np.std(v)) if v else 0
            table6[label] = {"mean": m, "std": s, "n": len(v)}
            print(f"  {desc:<50} mean={m:.3f}, std={s:.3f}, n={len(v)}")

    # === Signal Rating ===
    print("\n" + "=" * 110)
    print("FINAL SIGNAL RATING (based on leakage-free tile_work prediction)")
    print("=" * 110)
    ratings = {}
    for sig in SIGNALS:
        tile_mean = table5.get(sig, {}).get("mean", 0)
        tile_std = table5.get(sig, {}).get("std", 0)
        vis_leak = table1.get(sig, {}).get("future_vis", 0)

        if tile_mean > 0.3 and tile_std < 0.1:
            rating = "STRONG"
        elif tile_mean > 0.15 and tile_std < 0.15:
            rating = "MODERATE"
        elif tile_mean > 0.05:
            rating = "WEAK"
        else:
            rating = "DROP"

        ratings[sig] = {
            "tile_work_mean": tile_mean,
            "tile_work_std": tile_std,
            "vis_leak_pearson": vis_leak,
            "rating": rating,
        }
        print(f"  {sig:<25} tile_mean={tile_mean:.3f}, std={tile_std:.3f}, "
              f"vis_leak={vis_leak:.3f} → {rating}")

    # === Decision ===
    print("\n" + "=" * 110)
    print("DECISION")
    print("=" * 110)

    vis_tile = table5.get("visibility_count", {}).get("mean", 0)
    vis_std = table5.get("visibility_count", {}).get("std", 0)
    vis_leak_val = table6.get("test6", {}).get("mean", 0)
    vis_residual = table6.get("test3_tile", {}).get("mean", 0)
    screen_tile = table5.get("screen_radius_mean", {}).get("mean", 0)
    screen_std = table5.get("screen_radius_mean", {}).get("std", 0)

    print(f"\n  visibility_count → tile_work:    {vis_tile:.3f} (std={vis_std:.3f})")
    print(f"  visibility_count → vis (LEAK):   {vis_leak_val:.3f}")
    print(f"  visibility → residual tile_work: {vis_residual:.3f}")
    print(f"  screen_radius → tile_work:       {screen_tile:.3f} (std={screen_std:.3f})")

    # Gate checks
    gates = {}
    gates["A_visibility_predicts_work"] = vis_tile > 0.1
    gates["B_survives_leakage"] = vis_residual > 0.02  # Still meaningful after removing future_vis
    gates["C_cross_scene"] = vis_std < 0.1
    gates["D_mature_gaussians"] = table4.get("100-500", {}).get("tile_work_corr", 0) > 0.1 or \
                                  table4.get("500-2000", {}).get("tile_work_corr", 0) > 0.1 or \
                                  table4.get(">2000", {}).get("tile_work_corr", 0) > 0.1
    gates["E_screen_radius_better"] = screen_tile > vis_tile

    for g, v in gates.items():
        print(f"  Gate {g}: {'✅' if v else '❌'}")

    if all([gates["A_visibility_predicts_work"], gates["B_survives_leakage"], gates["C_cross_scene"]]):
        if gates["E_screen_radius_better"]:
            decision = "MODIFY — screen_radius_mean outperforms visibility for actual work; define exact utility domain"
        else:
            decision = "KEEP — visibility predicts actual work beyond leakage"
    elif vis_tile > 0.05 and gates["E_screen_radius_better"]:
        decision = "MODIFY — visibility weak for actual work; screen_radius is the better predictor"
    else:
        decision = "DROP — visibility loses predictive power against actual work"

    print(f"\n  DECISION: {decision}")

    # Save
    final = {
        "table1_leakage_free": table1,
        "table2_work_type": table2 if intersection else {},
        "table3_horizon": table3 if horizon else {},
        "table4_age": table4 if age else {},
        "table5_cross_scene": table5,
        "table6_leakage_control": table6 if leak else {},
        "ratings": ratings,
        "gates": gates,
        "decision": decision,
    }
    out_file = OUT_DIR / "final_comparison.json"
    with open(out_file, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
