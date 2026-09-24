#!/usr/bin/env python3
"""
Phase C53-Validation — Temporal Horizon Analysis

How does prediction of ACTUAL WORK change with Δ=10/50/100?
Compare visibility vs screen_radius vs ema_grad across horizons.
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
TARGETS = {
    "future_tile_work": "tiles_mean",
    "future_pixel_work": "area_mean",
    "future_update": "update_xyz",
    "future_grad": "grad_sum",
    "future_vis": "vis_count",
}


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def analyze_scene(scene, suffix=""):
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        return None
    data = np.load(filepath, allow_pickle=True)

    results = {"scene": scene + suffix}
    for d in DELTAS:
        d_results = {}
        for tname, tkey in TARGETS.items():
            sig_vals = {sig: [] for sig in SIGNALS}
            for cp in CHECKPOINTS:
                prefix = f"cp{cp}"
                d_prefix = f"{prefix}_d{d}"
                if f"{d_prefix}_alive" not in data:
                    continue
                alive = data[f"{d_prefix}_alive"]
                tkey_full = f"{d_prefix}_{tkey}"
                if tkey_full not in data:
                    continue
                target_vals = data[tkey_full].astype(np.float64)

                for sig in SIGNALS:
                    sig_key = f"{prefix}_s_{sig}"
                    if sig_key in data:
                        sv = data[sig_key].astype(np.float64)
                        if np.std(sv[alive]) > 1e-12 and np.std(target_vals[alive]) > 1e-12:
                            sig_vals[sig].append(safe_pearson(sv[alive], target_vals[alive]))

            d_results[tname] = {sig: float(np.mean(v)) if v else 0 for sig, v in sig_vals.items()}
        results[f"delta_{d}"] = d_results
    return results


def main():
    print("=" * 110)
    print("Phase C53-Validation — Temporal Horizon Analysis")
    print("=" * 110)

    scenes = ["room", "garden", "bicycle", "room_seed123", "garden_seed123"]
    all_results = {}

    for scene in scenes:
        parts = scene.split("_seed")
        base_scene = parts[0]
        suffix = f"_seed{parts[1]}" if len(parts) > 1 else ""
        results = analyze_scene(base_scene, suffix)
        if results:
            all_results[scene] = results

    # Cross-scene summary
    primary = ["room", "garden", "bicycle"]
    print("\n" + "=" * 110)
    print("CROSS-SCENE HORIZON (mean across room, garden, bicycle)")
    print("=" * 110)

    for tname in TARGETS:
        print(f"\n  Target: {tname}")
        print(f"  {'Signal':<25} {'Δ=10':>10} {'Δ=50':>10} {'Δ=100':>10} {'Trend':>10}")
        for sig in SIGNALS:
            vals = []
            for d in DELTAS:
                v = []
                for scene in primary:
                    if scene in all_results:
                        key = f"delta_{d}"
                        if key in all_results[scene] and tname in all_results[scene][key] and sig in all_results[scene][key][tname]:
                            v.append(all_results[scene][key][tname][sig])
                vals.append(float(np.mean(v)) if v else 0)
            trend = "↑" if vals[2] > vals[0] else "↓" if vals[2] < vals[0] else "→"
            print(f"  {sig:<25} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {trend:>10}")

    # Save
    out_file = OUT_DIR / "horizon_analysis.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
