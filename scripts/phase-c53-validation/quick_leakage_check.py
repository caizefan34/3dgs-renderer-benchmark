#!/usr/bin/env python3
"""
Quick leakage check: Compare visibility_count's prediction of:
  - future_vis_count (self-prediction, potentially leaked)
  - future_tiles_mean (actual rasterization work)
  - future_area_mean (actual pixel rendering work)
  - future_update_xyz (actual parameter update)

Using existing C53-Discovery raw data.
"""
import numpy as np
from pathlib import Path
import json

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery")
CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
DELTAS = [10, 50, 100]

SIGNALS = ["prev_grad_norm", "ema_grad_norm", "visibility_count",
           "screen_radius_mean", "opacity", "scale_norm", "age"]

# Targets to compare
TARGETS = {
    "future_vis_count": "vis_count",        # Self-predictive (leakage concern)
    "future_tiles_mean": "tiles_mean",       # Actual rasterization work
    "future_area_mean": "area_mean",         # Actual pixel rendering work
    "future_update_xyz": "update_xyz",       # Actual parameter update
    "future_grad_sum": "grad_sum",           # Future gradient
}


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def quick_check(scene, suffix=""):
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        print(f"  {filepath} not found")
        return None

    data = np.load(filepath, allow_pickle=True)
    results = {}

    for cp in CHECKPOINTS:
        prefix = f"cp{cp}"
        if f"{prefix}_ids" not in data:
            continue

        # Get signals
        signals = {}
        for sig in SIGNALS:
            key = f"{prefix}_s_{sig}"
            if key in data:
                signals[sig] = data[key].astype(np.float64)

        for d in DELTAS:
            d_prefix = f"{prefix}_d{d}"
            if f"{d_prefix}_alive" not in data:
                continue
            alive = data[f"{d_prefix}_alive"]

            for tname, tkey in TARGETS.items():
                tkey_full = f"{d_prefix}_{tkey}"
                if tkey_full not in data:
                    continue
                target_vals = data[tkey_full].astype(np.float64)

                for sig, sig_vals in signals.items():
                    mask = alive.copy()
                    if np.std(sig_vals[mask]) < 1e-12:
                        continue
                    p = safe_pearson(sig_vals[mask], target_vals[mask])
                    key = f"{sig}|{tname}|d{d}"
                    if key not in results:
                        results[key] = []
                    results[key].append(p)

    # Compute means
    summary = {}
    for key, vals in results.items():
        summary[key] = {
            "mean_pearson": float(np.mean(vals)),
            "std_pearson": float(np.std(vals)),
            "n": len(vals),
        }

    return summary


def main():
    print("=" * 100)
    print("QUICK LEAKAGE CHECK: Does visibility predict ACTUAL WORK or just future visibility?")
    print("=" * 100)

    all_results = {}
    for scene in ["room", "garden", "bicycle"]:
        print(f"\n--- {scene} ---")
        s = quick_check(scene)
        if s:
            all_results[scene] = s
            # Print key comparisons
            for d in DELTAS:
                print(f"\n  Δ={d}:")
                print(f"  {'Signal':<25} {'→ vis_count':>12} {'→ tiles_mean':>14} {'→ area_mean':>14} {'→ update_xyz':>14} {'→ grad_sum':>12}")
                for sig in ["visibility_count", "ema_grad_norm", "opacity", "prev_grad_norm"]:
                    vc = s.get(f"{sig}|future_vis_count|d{d}", {}).get("mean_pearson", 0)
                    tm = s.get(f"{sig}|future_tiles_mean|d{d}", {}).get("mean_pearson", 0)
                    am = s.get(f"{sig}|future_area_mean|d{d}", {}).get("mean_pearson", 0)
                    ux = s.get(f"{sig}|future_update_xyz|d{d}", {}).get("mean_pearson", 0)
                    gs = s.get(f"{sig}|future_grad_sum|d{d}", {}).get("mean_pearson", 0)
                    print(f"  {sig:<25} {vc:>12.3f} {tm:>14.3f} {am:>14.3f} {ux:>14.3f} {gs:>12.3f}")

    # Cross-scene summary
    print("\n\n=== CROSS-SCENE SUMMARY (mean across all scenes, checkpoints, deltas) ===")
    print(f"{'Signal':<25} {'→ vis_count':>12} {'→ tiles_mean':>14} {'→ area_mean':>14} {'→ update_xyz':>14} {'→ grad_sum':>12}")
    for sig in SIGNALS:
        vals = {}
        for tname in TARGETS:
            all_p = []
            for scene in all_results:
                for key, v in all_results[scene].items():
                    if key.startswith(f"{sig}|{tname}|"):
                        all_p.append(v["mean_pearson"])
            vals[tname] = float(np.mean(all_p)) if all_p else 0
        print(f"{sig:<25} {vals['future_vis_count']:>12.3f} {vals['future_tiles_mean']:>14.3f} "
              f"{vals['future_area_mean']:>14.3f} {vals['future_update_xyz']:>14.3f} {vals['future_grad_sum']:>12.3f}")

    # The critical comparison
    print("\n\n=== CRITICAL LEAKAGE TEST ===")
    for d in DELTAS:
        vc_p = []
        tm_p = []
        am_p = []
        for scene in all_results:
            for key, v in all_results[scene].items():
                if key == f"visibility_count|future_vis_count|d{d}":
                    vc_p.append(v["mean_pearson"])
                if key == f"visibility_count|future_tiles_mean|d{d}":
                    tm_p.append(v["mean_pearson"])
                if key == f"visibility_count|future_area_mean|d{d}":
                    am_p.append(v["mean_pearson"])
        if vc_p and tm_p:
            print(f"  Δ={d}: vis→vis={np.mean(vc_p):.3f}, vis→tiles={np.mean(tm_p):.3f}, "
                  f"vis→area={np.mean(am_p):.3f}, "
                  f"leakage_ratio={np.mean(tm_p)/np.mean(vc_p):.2f}")

    # Save
    out_file = RESULT_DIR / "quick_leakage_check.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
