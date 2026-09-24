#!/usr/bin/env python3
"""
Phase C53-Validation — Backward Work Analysis

Analyzes backward work = tiles × grad_norm product.
For existing C53-Discovery data, we can approximate backward work as:
  backward_work_proxy = tiles_mean × grad_mean (element-wise product per Gaussian)

This is a PROXY because true backward work requires per-iteration multiplication.
The per-iteration product is NOT the same as the product of sums.

However, we can still test whether signals predict this proxy.
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

    results = {"scene": scene + suffix, "checkpoints": {}}

    for cp in CHECKPOINTS:
        prefix = f"cp{cp}"
        if f"{prefix}_ids" not in data:
            continue

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

            # Get component targets
            tiles_mean = data[f"{d_prefix}_tiles_mean"].astype(np.float64) if f"{d_prefix}_tiles_mean" in data else None
            grad_mean = data[f"{d_prefix}_grad_mean"].astype(np.float64) if f"{d_prefix}_grad_mean" in data else None
            vis_count = data[f"{d_prefix}_vis_count"].astype(np.float64) if f"{d_prefix}_vis_count" in data else None
            update_xyz = data[f"{d_prefix}_update_xyz"].astype(np.float64) if f"{d_prefix}_update_xyz" in data else None

            if tiles_mean is None or grad_mean is None:
                continue

            # Construct backward work proxy: tiles × grad (element-wise)
            # NOTE: This is tiles_mean × grad_mean, NOT Σ(tiles(τ)×grad(τ))
            # It's a proxy. True per-iteration product requires new collection.
            backward_work_proxy = tiles_mean * grad_mean

            # Also construct: vis × grad (visibility-weighted backward)
            vis_grad_proxy = vis_count * grad_mean if vis_count is not None else None

            d_results = {}
            for target_name, target_vals in [
                ("backward_work_proxy", backward_work_proxy),
                ("vis_grad_proxy", vis_grad_proxy) if vis_grad_proxy is not None else ("skip", None),
                ("tiles_mean", tiles_mean),
                ("grad_mean", grad_mean),
                ("update_xyz", update_xyz) if update_xyz is not None else ("skip", None),
            ]:
                if target_name == "skip" or target_vals is None:
                    continue
                sig_results = {}
                for sig, sig_vals in signals.items():
                    if np.std(sig_vals[alive]) < 1e-12 or np.std(target_vals[alive]) < 1e-12:
                        continue
                    sig_results[sig] = {
                        "pearson": safe_pearson(sig_vals[alive], target_vals[alive]),
                        "recall_20": recall_at_k(sig_vals[alive], target_vals[alive], 20),
                    }
                d_results[target_name] = sig_results
            cp_results[str(d)] = d_results
        results["checkpoints"][str(cp)] = cp_results

    return results


def main():
    print("=" * 110)
    print("Phase C53-Validation — Backward Work Analysis (Proxy from Existing Data)")
    print("NOTE: backward_work_proxy = tiles_mean × grad_mean (element-wise product of means)")
    print("      This is NOT the true Σ(tiles(τ)×grad(τ)). True value needs new collection.")
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
    print("CROSS-SCENE SUMMARY (mean across room, garden, bicycle, Δ=50)")
    print("=" * 110)

    for target_name in ["backward_work_proxy", "vis_grad_proxy", "tiles_mean", "grad_mean", "update_xyz"]:
        print(f"\n  Target: {target_name}")
        print(f"  {'Signal':<25} {'Pearson':>10} {'Recall@20':>10}")
        for sig in SIGNALS:
            pears = []
            recs = []
            for scene in primary:
                if scene in all_results:
                    for cp in CHECKPOINTS:
                        cp_str = str(cp)
                        if cp_str in all_results[scene].get("checkpoints", {}):
                            cp_data = all_results[scene]["checkpoints"][cp_str]
                            if "50" in cp_data and target_name in cp_data["50"] and sig in cp_data["50"][target_name]:
                                pears.append(cp_data["50"][target_name][sig]["pearson"])
                                recs.append(cp_data["50"][target_name][sig]["recall_20"])
            if pears:
                print(f"  {sig:<25} {np.mean(pears):>10.3f} {np.mean(recs):>10.3f}")

    # Key comparison
    print("\n\n" + "=" * 110)
    print("KEY COMPARISON: visibility vs ema_grad for backward work proxy")
    print("=" * 110)
    for d in DELTAS:
        vis_p, ema_p = [], []
        for scene in primary:
            if scene in all_results:
                for cp in CHECKPOINTS:
                    cp_str = str(cp)
                    if cp_str in all_results[scene].get("checkpoints", {}):
                        cp_data = all_results[scene]["checkpoints"][cp_str]
                        d_str = str(d)
                        if d_str in cp_data and "backward_work_proxy" in cp_data[d_str]:
                            if "visibility_count" in cp_data[d_str]["backward_work_proxy"]:
                                vis_p.append(cp_data[d_str]["backward_work_proxy"]["visibility_count"]["pearson"])
                            if "ema_grad_norm" in cp_data[d_str]["backward_work_proxy"]:
                                ema_p.append(cp_data[d_str]["backward_work_proxy"]["ema_grad_norm"]["pearson"])
        if vis_p and ema_p:
            print(f"  Δ={d}: visibility→backward_work={np.mean(vis_p):.3f}, ema_grad→backward_work={np.mean(ema_p):.3f}")

    # Save
    out_file = OUT_DIR / "backward_work_analysis.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
