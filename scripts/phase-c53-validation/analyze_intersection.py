#!/usr/bin/env python3
"""
Phase C53-Validation — Intersection-Level Analysis

Analyzes which layer visibility predicts:
  - intersection (tile intersection count = tiles_mean)
  - rasterization (pixel work = area_mean)
  - backward (update = update_xyz, grad = grad_sum)

Compares visibility vs screen_radius vs scale for each layer.
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


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def recall_at_k(signal_vals, target_vals, k_percent):
    n = len(signal_vals)
    k = max(1, int(n * k_percent / 100))
    top_signal = set(np.argsort(signal_vals)[-k:])
    top_target = set(np.argsort(target_vals)[-k:])
    if len(top_target) == 0:
        return 0.0
    return len(top_signal & top_target) / len(top_target)


def analyze_scene(scene, suffix=""):
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        return None
    data = np.load(filepath, allow_pickle=True)

    results = {"scene": scene + suffix}

    # For each Δ, compute mean across checkpoints
    for d in DELTAS:
        # Layer targets
        layer_targets = {
            "intersection_tiles": "tiles_mean",   # Tile intersection count
            "rasterization_area": "area_mean",     # Pixel rendering work
            "backward_update": "update_xyz",       # Parameter update
            "backward_grad": "grad_sum",           # Gradient flow
            "visibility": "vis_count",             # Self-predictive (for comparison)
        }

        d_results = {}
        for layer_name, tkey in layer_targets.items():
            sig_results = {}
            for sig in SIGNALS:
                vals = []
                recalls = []
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
                    sig_key = f"{prefix}_s_{sig}"
                    if sig_key not in data:
                        continue
                    sv = data[sig_key].astype(np.float64)
                    if np.std(sv[alive]) > 1e-12 and np.std(target_vals[alive]) > 1e-12:
                        vals.append(safe_pearson(sv[alive], target_vals[alive]))
                        recalls.append(recall_at_k(sv[alive], target_vals[alive], 20))
                if vals:
                    sig_results[sig] = {
                        "pearson": float(np.mean(vals)),
                        "recall_20": float(np.mean(recalls)),
                        "n": len(vals),
                    }
            d_results[layer_name] = sig_results
        results[f"delta_{d}"] = d_results

    return results


def main():
    print("=" * 110)
    print("Phase C53-Validation — Intersection-Level Analysis")
    print("Which layer does each signal predict?")
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
    layers = ["intersection_tiles", "rasterization_area", "backward_update", "backward_grad", "visibility"]

    for d in DELTAS:
        print(f"\n{'='*110}")
        print(f"CROSS-SCENE SUMMARY (mean across room, garden, bicycle, Δ={d})")
        print(f"{'='*110}")
        print(f"  {'Signal':<25} {'→ tiles':>10} {'→ area':>10} {'→ update':>10} {'→ grad':>10} {'→ vis':>10}")
        for sig in SIGNALS:
            vals = {}
            for layer in layers:
                v = []
                for scene in primary:
                    if scene in all_results:
                        key = f"delta_{d}"
                        if key in all_results[scene] and layer in all_results[scene][key] and sig in all_results[scene][key][layer]:
                            v.append(all_results[scene][key][layer][sig]["pearson"])
                vals[layer] = float(np.mean(v)) if v else 0
            print(f"  {sig:<25} {vals['intersection_tiles']:>10.3f} {vals['rasterization_area']:>10.3f} "
                  f"{vals['backward_update']:>10.3f} {vals['backward_grad']:>10.3f} {vals['visibility']:>10.3f}")

    # Save
    out_file = OUT_DIR / "intersection_work_analysis.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
