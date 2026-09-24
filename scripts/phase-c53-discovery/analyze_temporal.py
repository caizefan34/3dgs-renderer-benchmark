#!/usr/bin/env python3
"""
Phase C53-Discovery — Temporal Horizon Analysis

For each signal, measure how correlation decays with prediction horizon Δ.
Output: temporal_horizon.json
"""
import json
from pathlib import Path
import numpy as np

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery")

SIGNALS = [
    "prev_grad_norm", "ema_grad_norm", "visibility_count",
    "screen_radius_mean", "opacity", "scale_norm", "age",
]
DELTAS = [10, 50, 100]
CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]

# Primary utility targets for temporal analysis
UTILITIES = ["grad_sum", "update_xyz", "vis_count"]


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_scene(scene, suffix=""):
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        return None

    data = np.load(filepath, allow_pickle=True)
    results = {"scene": scene, "checkpoints": {}}

    for cp in CHECKPOINTS:
        prefix = f"cp{cp}"
        if f"{prefix}_ids" not in data:
            continue

        signals = {}
        for sig in SIGNALS:
            key = f"{prefix}_s_{sig}"
            if key in data:
                signals[sig] = data[key].astype(np.float64)

        cp_results = {"iter": cp, "horizons": {}}

        for util in UTILITIES:
            cp_results["horizons"][util] = {}
            for d in DELTAS:
                d_prefix = f"{prefix}_d{d}"
                util_key = f"{d_prefix}_{util}"
                alive_key = f"{d_prefix}_alive"
                if util_key not in data or alive_key not in data:
                    continue

                util_vals = data[util_key].astype(np.float64)
                alive = data[alive_key]
                mask = alive.copy()

                cp_results["horizons"][util][f"delta_{d}"] = {}
                for sig, sig_vals in signals.items():
                    if np.std(sig_vals[mask]) < 1e-12:
                        continue
                    cp_results["horizons"][util][f"delta_{d}"][sig] = {
                        "pearson": safe_pearson(sig_vals[mask], util_vals[mask]),
                        "spearman": safe_spearman(sig_vals[mask], util_vals[mask]),
                    }

        results["checkpoints"][str(cp)] = cp_results

    # Summary: mean Pearson per signal × utility × delta across checkpoints
    summary = {"temporal_decay": {}}
    for util in UTILITIES:
        summary["temporal_decay"][util] = {}
        for sig in SIGNALS:
            summary["temporal_decay"][util][sig] = {}
            for d in DELTAS:
                pearsons = []
                for cp_key, cp_data in results["checkpoints"].items():
                    if util in cp_data["horizons"] and f"delta_{d}" in cp_data["horizons"][util]:
                        if sig in cp_data["horizons"][util][f"delta_{d}"]:
                            pearsons.append(cp_data["horizons"][util][f"delta_{d}"][sig]["pearson"])
                if pearsons:
                    summary["temporal_decay"][util][sig][f"delta_{d}"] = {
                        "mean_pearson": float(np.mean(pearsons)),
                        "std_pearson": float(np.std(pearsons)),
                        "n": len(pearsons),
                    }

    results["summary"] = summary
    return results


def main():
    all_results = {}
    for scene in ["room", "garden", "bicycle"]:
        result = analyze_scene(scene)
        if result:
            all_results[scene] = result
        result_ctrl = analyze_scene(scene, "_seed123")
        if result_ctrl:
            all_results[f"{scene}_seed123"] = result_ctrl

    out_file = RESULT_DIR / "temporal_horizon.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
