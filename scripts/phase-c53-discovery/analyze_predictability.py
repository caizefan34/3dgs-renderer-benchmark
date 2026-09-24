#!/usr/bin/env python3
"""
Phase C53-Discovery — Analyze Predictability

Core analysis: for each signal S × utility target U × Δ, compute:
  - Pearson correlation
  - Spearman correlation
  - Recall@K (1%, 5%, 10%, 20%, 50%)
  - Coverage@K

Outputs: room_signals.json, garden_signals.json, bicycle_signals.json,
         future_gradient.json, future_update.json, future_render_proxy.json
"""
import json, sys
from pathlib import Path
import numpy as np

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery")

SIGNALS = [
    "prev_grad_norm", "ema_grad_norm", "visibility_count",
    "screen_radius_mean", "opacity", "scale_norm", "age",
]

# Utility targets at each delta
UTILITY_TARGETS = {
    "future_gradient": {
        "keys": {"sum": "grad_sum", "max": "grad_max", "mean": "grad_mean"},
        "primary": "grad_sum",
    },
    "future_update": {
        "keys": {"xyz": "update_xyz", "opacity": "update_opacity", "scale": "update_scale",
                 "rot": "update_rot", "shs": "update_shs"},
        "primary": "update_xyz",
    },
    "future_render_proxy": {
        "keys": {"vis_count": "vis_count", "radius_mean": "radius_mean",
                 "area_mean": "area_mean", "tiles_mean": "tiles_mean"},
        "primary": "vis_count",
    },
}

DELTAS = [10, 50, 100]
KS = [1, 5, 10, 20, 50]
CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]


def safe_pearson(x, y):
    """Compute Pearson correlation, return 0 if undefined."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    """Compute Spearman correlation via rank substitution."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def recall_at_k(signal, utility, k_percent):
    """Recall@K: fraction of top-K% utility Gaussians in top-K% signal Gaussians."""
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_util = set(np.argsort(utility)[-k:])
    top_sig = set(np.argsort(signal)[-k:])
    return len(top_util & top_sig) / len(top_util) if top_util else 0.0


def coverage_at_k(signal, utility, k_percent):
    """Coverage@K: fraction of total utility covered by top-K% signal Gaussians."""
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_sig = np.argsort(signal)[-k:]
    total_u = utility.sum()
    if total_u < 1e-12:
        return 0.0
    return float(utility[top_sig].sum() / total_u)


def analyze_scene(scene, suffix=""):
    """Analyze one scene's raw data."""
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        print(f"  WARNING: {filepath} not found, skipping {scene}{suffix}")
        return None

    data = np.load(filepath, allow_pickle=True)
    results = {"scene": scene, "suffix": suffix, "checkpoints": {}}

    for cp in CHECKPOINTS:
        prefix = f"cp{cp}"
        if f"{prefix}_ids" not in data:
            continue

        n_total = int(data[f"{prefix}_n_total"])
        n_sample = len(data[f"{prefix}_ids"])

        # Get signals
        signals = {}
        for sig in SIGNALS:
            key = f"{prefix}_s_{sig}"
            if key in data:
                signals[sig] = data[key].astype(np.float64)

        # Get outcomes
        outcome_key = f"{prefix}_outcome"
        outcomes = data[outcome_key] if outcome_key in data else np.zeros(n_sample, dtype=np.int8)

        cp_results = {"iter": cp, "n_sample": n_sample, "n_total": n_total,
                       "signals": {}, "utilities": {}}

        # Signal statistics
        for sig, vals in signals.items():
            cp_results["signals"][sig] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "nonzero": int(np.count_nonzero(vals)),
            }

        # Outcome distribution
        cp_results["outcomes"] = {
            "unchanged": int((outcomes == 0).sum()),
            "cloned": int((outcomes == 1).sum()),
            "split": int((outcomes == 2).sum()),
            "pruned": int((outcomes == 3).sum()),
        }

        # For each delta
        for d in DELTAS:
            d_prefix = f"{prefix}_d{d}"
            if f"{d_prefix}_alive" not in data:
                continue

            alive = data[f"{d_prefix}_alive"]
            d_results = {"delta": d, "n_alive": int(alive.sum())}

            # Get utility targets
            utilities = {}
            for util_name, util_spec in UTILITY_TARGETS.items():
                for sub_name, key in util_spec["keys"].items():
                    full_key = f"{d_prefix}_{key}"
                    if full_key in data:
                        utilities[f"{util_name}_{sub_name}"] = data[full_key].astype(np.float64)

            # Compute correlations for each signal × utility
            correlations = {}
            for sig_name, sig_vals in signals.items():
                correlations[sig_name] = {}
                for util_name, util_vals in utilities.items():
                    # Only use alive Gaussians for correlation
                    mask = alive.copy()
                    if np.std(sig_vals[mask]) < 1e-12:
                        corr = {"pearson": 0.0, "spearman": 0.0}
                    else:
                        corr = {
                            "pearson": safe_pearson(sig_vals[mask], util_vals[mask]),
                            "spearman": safe_spearman(sig_vals[mask], util_vals[mask]),
                        }

                    # Recall@K and Coverage@K
                    recall = {}
                    coverage = {}
                    for k in KS:
                        recall[f"recall_{k}"] = recall_at_k(sig_vals[mask], util_vals[mask], k)
                        coverage[f"coverage_{k}"] = coverage_at_k(sig_vals[mask], util_vals[mask], k)

                    correlations[sig_name][util_name] = {
                        **corr, **{f"recall_{k}": recall[f"recall_{k}"] for k in KS},
                        **{f"coverage_{k}": coverage[f"coverage_{k}"] for k in KS},
                    }

            d_results["correlations"] = correlations
            d_results["utility_stats"] = {
                u: {"mean": float(v[alive].mean()) if alive.any() else 0,
                     "std": float(v[alive].std()) if alive.any() else 0,
                     "nonzero": int(np.count_nonzero(v[alive])) if alive.any() else 0}
                for u, v in utilities.items()
            }
            cp_results["utilities"][f"delta_{d}"] = d_results

        results["checkpoints"][str(cp)] = cp_results

    # Compute summary across checkpoints
    summary = compute_summary(results)
    results["summary"] = summary
    return results


def compute_summary(results):
    """Compute mean correlations across all checkpoints for each signal × utility × delta."""
    summary = {"mean_correlations": {}}

    for d in DELTAS:
        d_key = f"delta_{d}"
        for sig in SIGNALS:
            for util_name in [f"future_gradient_sum", f"future_gradient_max",
                              f"future_update_xyz", f"future_update_opacity",
                              f"future_update_scale", f"future_update_rot",
                              f"future_update_shs",
                              f"future_render_proxy_vis_count",
                              f"future_render_proxy_radius_mean",
                              f"future_render_proxy_area_mean",
                              f"future_render_proxy_tiles_mean"]:
                pearsons = []
                spearmans = []
                recalls = {k: [] for k in KS}
                coverages = {k: [] for k in KS}

                for cp_key, cp_data in results["checkpoints"].items():
                    if d_key not in cp_data["utilities"]:
                        continue
                    corr = cp_data["utilities"][d_key]["correlations"]
                    if sig in corr and util_name in corr[sig]:
                        c = corr[sig][util_name]
                        pearsons.append(c["pearson"])
                        spearmans.append(c["spearman"])
                        for k in KS:
                            recalls[k].append(c[f"recall_{k}"])
                            coverages[k].append(c[f"coverage_{k}"])

                if pearsons:
                    key = f"{sig}|{util_name}|d{d}"
                    summary["mean_correlations"][key] = {
                        "pearson_mean": float(np.mean(pearsons)),
                        "pearson_std": float(np.std(pearsons)),
                        "spearman_mean": float(np.mean(spearmans)),
                        "n_checkpoints": len(pearsons),
                        **{f"recall_{k}_mean": float(np.mean(recalls[k])) for k in KS if recalls[k]},
                        **{f"coverage_{k}_mean": float(np.mean(coverages[k])) for k in KS if coverages[k]},
                    }

    return summary


def main():
    scenes = ["room", "garden", "bicycle"]
    all_results = {}

    for scene in scenes:
        # Primary seed
        result = analyze_scene(scene)
        if result:
            all_results[scene] = result
            out_file = RESULT_DIR / f"{scene}_signals.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved {out_file}")

        # Control seed (if exists)
        result_ctrl = analyze_scene(scene, "_seed123")
        if result_ctrl:
            all_results[f"{scene}_seed123"] = result_ctrl
            out_file = RESULT_DIR / f"{scene}_seed123_signals.json"
            with open(out_file, 'w') as f:
                json.dump(result_ctrl, f, indent=2)
            print(f"  Saved {out_file}")

    # Generate utility-specific analysis files
    for util_name, util_spec in UTILITY_TARGETS.items():
        util_results = {}
        for scene in scenes:
            if scene in all_results:
                util_results[scene] = {}
                for cp_key, cp_data in all_results[scene]["checkpoints"].items():
                    util_results[scene][cp_key] = {}
                    for d in DELTAS:
                        d_key = f"delta_{d}"
                        if d_key in cp_data["utilities"]:
                            corr = cp_data["utilities"][d_key]["correlations"]
                            util_results[scene][cp_key][f"delta_{d}"] = {
                                sig: corr.get(sig, {}).get(f"{util_name}_{sub}", {})
                                for sig in SIGNALS
                                for sub in util_spec["keys"]
                            }

        out_file = RESULT_DIR / f"{util_name}.json"
        with open(out_file, 'w') as f:
            json.dump(util_results, f, indent=2)
        print(f"  Saved {out_file}")

    print("\nPredictability analysis complete.")


if __name__ == "__main__":
    main()
