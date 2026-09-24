#!/usr/bin/env python3
"""
Phase C53-Validation — Age-Stratified Analysis

For each age group, measure visibility → future actual work (NOT future visibility).
Tests whether the "Gaussian maturation" effect survives with leakage-free targets.
"""
import json
from pathlib import Path
import numpy as np

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery")
OUT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
DELTAS = [10, 50, 100]
SIGNALS = ["prev_grad_norm", "ema_grad_norm", "visibility_count",
           "screen_radius_mean", "opacity", "scale_norm", "age"]
AGE_GROUPS = [(0, 100), (100, 500), (500, 2000), (2000, 10**9)]
AGE_LABELS = ["<100", "100-500", "500-2000", ">2000"]

TARGETS = {
    "future_tile_work": "tiles_mean",
    "future_pixel_work": "area_mean",
    "future_update": "update_xyz",
    "future_grad": "grad_sum",
    "future_vis": "vis_count",
}


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def analyze_scene(scene, suffix=""):
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        return None

    data = np.load(filepath, allow_pickle=True)
    results = {"scene": scene + suffix, "age_groups": {}}

    for ag_idx, (ag_lo, ag_hi) in enumerate(AGE_GROUPS):
        ag_label = AGE_LABELS[ag_idx]
        ag_results = {"label": ag_label, "checkpoints": {}}

        for cp in CHECKPOINTS:
            prefix = f"cp{cp}"
            if f"{prefix}_ids" not in data:
                continue

            age_vals = data[f"{prefix}_s_age"].astype(np.float64)
            age_mask = (age_vals >= ag_lo) & (age_vals < ag_hi)

            signals = {}
            for sig in SIGNALS:
                key = f"{prefix}_s_{sig}"
                if key in data:
                    signals[sig] = data[key].astype(np.float64)

            cp_results = {}
            for d in DELTAS:
                d_prefix = f"{prefix}_d{d}"
                if f"{d_prefix}_alive" not in data:
                    continue
                alive = data[f"{d_prefix}_alive"]
                mask = alive & age_mask

                d_results = {}
                for tname, tkey in TARGETS.items():
                    tkey_full = f"{d_prefix}_{tkey}"
                    if tkey_full not in data:
                        continue
                    target_vals = data[tkey_full].astype(np.float64)

                    sig_results = {}
                    for sig, sig_vals in signals.items():
                        if mask.sum() < 10:
                            continue
                        sv = sig_vals[mask]
                        tv = target_vals[mask]
                        if np.std(sv) < 1e-12 or np.std(tv) < 1e-12:
                            continue
                        sig_results[sig] = safe_pearson(sv, tv)
                    d_results[tname] = sig_results
                cp_results[str(d)] = d_results
            ag_results["checkpoints"][str(cp)] = cp_results
        results["age_groups"][ag_label] = ag_results

    return results


def main():
    print("=" * 110)
    print("Phase C53-Validation — Age-Stratified Analysis (Leakage-Free Targets)")
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

    # Print summary for primary scenes, Δ=50
    primary = ["room", "garden", "bicycle"]
    print("\n\n" + "=" * 110)
    print("AGE-STRATIFIED SUMMARY (mean across room, garden, bicycle, Δ=50)")
    print("=" * 110)

    for ag_label in AGE_LABELS:
        print(f"\n  Age {ag_label}:")
        print(f"  {'Signal':<25} {'→ tile_work':>14} {'→ pixel_work':>14} {'→ update':>10} {'→ grad':>10} {'→ vis(LEAK)':>12}")
        for sig in SIGNALS:
            vals = {t: [] for t in TARGETS}
            n_vals = 0
            for scene in primary:
                if scene in all_results and ag_label in all_results[scene]["age_groups"]:
                    ag_data = all_results[scene]["age_groups"][ag_label]
                    for cp in CHECKPOINTS:
                        cp_str = str(cp)
                        if cp_str in ag_data.get("checkpoints", {}):
                            cp_data = ag_data["checkpoints"][cp_str]
                            if "50" in cp_data:
                                for tname in TARGETS:
                                    if tname in cp_data["50"] and sig in cp_data["50"][tname]:
                                        vals[tname].append(cp_data["50"][tname][sig])
                                        if tname == "future_tile_work":
                                            n_vals += 1
            means = {t: float(np.mean(v)) if v else 0 for t, v in vals.items()}
            print(f"  {sig:<25} {means['future_tile_work']:>14.3f} "
                  f"{means['future_pixel_work']:>14.3f} {means['future_update']:>10.3f} "
                  f"{means['future_grad']:>10.3f} {means['future_vis']:>12.3f}")

    # Best signal per age group
    print("\n\n" + "=" * 110)
    print("BEST SIGNAL PER AGE GROUP (Δ=50, target=tile_work)")
    print("=" * 110)
    for ag_label in AGE_LABELS:
        best_sig, best_val = "none", -999
        for sig in SIGNALS:
            vals = []
            for scene in primary:
                if scene in all_results and ag_label in all_results[scene]["age_groups"]:
                    ag_data = all_results[scene]["age_groups"][ag_label]
                    for cp in CHECKPOINTS:
                        cp_str = str(cp)
                        if cp_str in ag_data.get("checkpoints", {}):
                            cp_data = ag_data["checkpoints"][cp_str]
                            if "50" in cp_data and "future_tile_work" in cp_data["50"] and sig in cp_data["50"]["future_tile_work"]:
                                vals.append(cp_data["50"]["future_tile_work"][sig])
            if vals:
                m = float(np.mean(vals))
                if m > best_val:
                    best_val = m
                    best_sig = sig
        print(f"  Age {ag_label:<12}: best={best_sig:<25} Pearson={best_val:.3f}")

    # Save
    out_file = OUT_DIR / "age_analysis.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
