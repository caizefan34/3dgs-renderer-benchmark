#!/usr/bin/env python3
"""
Phase C53-Validation — Future Work Analysis

Analyzes existing C53-Discovery raw data with LEAKAGE-FREE targets:
  T1 future_tile_work    = tiles_mean (actual rasterization work)
  T2 future_backward_work = tiles × grad_norm product (needs new collection)
  T3 future_pixel_work   = area_mean (actual pixel rendering work)
  T4 future_update       = update_xyz (parameter update)

Signals (same 7 as C53-Discovery):
  S1-S7: prev_grad, ema_grad, visibility, screen_radius, opacity, scale, age

Outputs:
  - Pearson, Spearman, Recall@10/20/50 for each signal × target × Δ
  - Cross-scene comparison
"""
import json, sys
from pathlib import Path
import numpy as np

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery")
OUT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
DELTAS = [10, 50, 100]

SIGNALS = [
    "prev_grad_norm", "ema_grad_norm", "visibility_count",
    "screen_radius_mean", "opacity", "scale_norm", "age",
]

# Leakage-free targets (actual work, NOT future visibility)
TARGETS = {
    "future_tile_work": "tiles_mean",       # T1: actual rasterization work
    "future_pixel_work": "area_mean",        # T3: actual pixel rendering work
    "future_update": "update_xyz",           # T4: parameter update
    "future_grad": "grad_sum",               # Future gradient (secondary)
    "future_vis": "vis_count",               # Self-predictive target (for comparison)
}


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def recall_at_k(signal_vals, target_vals, k_percent):
    """Fraction of top-K% utility Gaussians in top-K% signal Gaussians."""
    n = len(signal_vals)
    k = max(1, int(n * k_percent / 100))
    top_signal = set(np.argsort(signal_vals)[-k:])
    top_target = set(np.argsort(target_vals)[-k:])
    if len(top_target) == 0:
        return 0.0
    return len(top_signal & top_target) / len(top_target)


def coverage_at_k(signal_vals, target_vals, k_percent):
    """Fraction of total utility covered by top-K% signal Gaussians."""
    n = len(signal_vals)
    k = max(1, int(n * k_percent / 100))
    top_signal = np.argsort(signal_vals)[-k:]
    total = target_vals.sum()
    if total < 1e-12:
        return 0.0
    return float(target_vals[top_signal].sum() / total)


def analyze_scene(scene, suffix=""):
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        print(f"  {filepath} not found")
        return None

    data = np.load(filepath, allow_pickle=True)
    results = {"scene": scene + suffix, "checkpoints": {}}

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

        cp_results = {}
        for d in DELTAS:
            d_prefix = f"{prefix}_d{d}"
            if f"{d_prefix}_alive" not in data:
                continue
            alive = data[f"{d_prefix}_alive"]

            d_results = {}
            for tname, tkey in TARGETS.items():
                tkey_full = f"{d_prefix}_{tkey}"
                if tkey_full not in data:
                    continue
                target_vals = data[tkey_full].astype(np.float64)

                sig_results = {}
                for sig, sig_vals in signals.items():
                    mask = alive.copy()
                    if np.std(sig_vals[mask]) < 1e-12:
                        continue
                    if np.std(target_vals[mask]) < 1e-12:
                        continue
                    sv = sig_vals[mask]
                    tv = target_vals[mask]

                    sig_results[sig] = {
                        "pearson": safe_pearson(sv, tv),
                        "spearman": safe_spearman(sv, tv),
                        "recall_10": recall_at_k(sv, tv, 10),
                        "recall_20": recall_at_k(sv, tv, 20),
                        "recall_50": recall_at_k(sv, tv, 50),
                        "coverage_20": coverage_at_k(sv, tv, 20),
                    }
                d_results[tname] = sig_results
            cp_results[d] = d_results
        results["checkpoints"][str(cp)] = cp_results

    return results


def compute_summary(scene_results):
    """Compute mean across checkpoints for each signal × target × delta."""
    summary = {"scene": scene_results["scene"], "deltas": {}}

    for d in DELTAS:
        d_summary = {}
        for tname in TARGETS:
            t_summary = {}
            for sig in SIGNALS:
                vals = {k: [] for k in ["pearson", "spearman", "recall_10", "recall_20", "recall_50", "coverage_20"]}
                for cp in CHECKPOINTS:
                    cp_str = str(cp)
                    if cp_str in scene_results.get("checkpoints", {}):
                        cp_data = scene_results["checkpoints"][cp_str]
                        if d in cp_data and tname in cp_data[d] and sig in cp_data[d][tname]:
                            for k in vals:
                                vals[k].append(cp_data[d][tname][sig][k])
                if vals["pearson"]:
                    t_summary[sig] = {k: float(np.mean(v)) for k, v in vals.items()}
            if t_summary:
                d_summary[tname] = t_summary
        if d_summary:
            summary["deltas"][d] = d_summary

    return summary


def main():
    print("=" * 110)
    print("Phase C53-Validation — Future Work Analysis (Leakage-Free Targets)")
    print("=" * 110)

    scenes = ["room", "garden", "bicycle", "room_seed123", "garden_seed123"]
    all_results = {}
    all_summaries = {}

    for scene in scenes:
        print(f"\n--- {scene} ---")
        parts = scene.split("_seed")
        base_scene = parts[0]
        suffix = f"_seed{parts[1]}" if len(parts) > 1 else ""
        results = analyze_scene(base_scene, suffix)
        if results:
            all_results[scene] = results
            summary = compute_summary(results)
            all_summaries[scene] = summary

            # Print key results for delta=50
            d = 50
            if d in summary.get("deltas", {}):
                print(f"\n  Δ={d}: Leakage-Free Pearson")
                print(f"  {'Signal':<25} {'→ tile_work':>14} {'→ pixel_work':>14} {'→ update':>10} {'→ grad':>10} {'→ vis(LEAK)':>12}")
                for sig in SIGNALS:
                    vals = {}
                    for tname in TARGETS:
                        if tname in summary["deltas"][d] and sig in summary["deltas"][d][tname]:
                            vals[tname] = summary["deltas"][d][tname][sig]["pearson"]
                        else:
                            vals[tname] = 0
                    print(f"  {sig:<25} {vals['future_tile_work']:>14.3f} "
                          f"{vals['future_pixel_work']:>14.3f} {vals['future_update']:>10.3f} "
                          f"{vals['future_grad']:>10.3f} {vals['future_vis']:>12.3f}")

    # Cross-scene summary (primary 3 scenes)
    print("\n\n" + "=" * 110)
    print("CROSS-SCENE SUMMARY (mean across room, garden, bicycle)")
    print("=" * 110)

    primary_scenes = ["room", "garden", "bicycle"]
    for d in DELTAS:
        print(f"\n  Δ={d}:")
        print(f"  {'Signal':<25} {'→ tile_work':>14} {'→ pixel_work':>14} {'→ update':>10} {'→ grad':>10} {'→ vis(LEAK)':>12}  {'tile_std':>10}")
        for sig in SIGNALS:
            vals = {t: [] for t in TARGETS}
            for scene in primary_scenes:
                if scene in all_summaries and d in all_summaries[scene].get("deltas", {}):
                    for tname in TARGETS:
                        if tname in all_summaries[scene]["deltas"][d] and sig in all_summaries[scene]["deltas"][d][tname]:
                            vals[tname].append(all_summaries[scene]["deltas"][d][tname][sig]["pearson"])
            means = {t: float(np.mean(v)) if v else 0 for t, v in vals.items()}
            stds = {t: float(np.std(v)) if v else 0 for t, v in vals.items()}
            print(f"  {sig:<25} {means['future_tile_work']:>14.3f} "
                  f"{means['future_pixel_work']:>14.3f} {means['future_update']:>10.3f} "
                  f"{means['future_grad']:>10.3f} {means['future_vis']:>12.3f}  {stds['future_tile_work']:>10.3f}")

    # Recall@K summary
    print("\n\n" + "=" * 110)
    print("CROSS-SCENE Recall@K (mean across room, garden, bicycle, Δ=50, target=tile_work)")
    print("=" * 110)
    print(f"  {'Signal':<25} {'Recall@10':>12} {'Recall@20':>12} {'Recall@50':>12} {'Coverage@20':>14}")
    for sig in SIGNALS:
        recalls = {k: [] for k in ["recall_10", "recall_20", "recall_50", "coverage_20"]}
        for scene in primary_scenes:
            if scene in all_summaries and 50 in all_summaries[scene].get("deltas", {}):
                d_data = all_summaries[scene]["deltas"][50]
                if "future_tile_work" in d_data and sig in d_data["future_tile_work"]:
                    for k in recalls:
                        recalls[k].append(d_data["future_tile_work"][sig][k])
        means = {k: float(np.mean(v)) if v else 0 for k, v in recalls.items()}
        print(f"  {sig:<25} {means['recall_10']:>12.3f} {means['recall_20']:>12.3f} "
              f"{means['recall_50']:>12.3f} {means['coverage_20']:>14.3f}")

    # Save
    out_file = OUT_DIR / "future_work_analysis.json"
    with open(out_file, 'w') as f:
        json.dump({"per_scene": all_results, "summaries": all_summaries}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
