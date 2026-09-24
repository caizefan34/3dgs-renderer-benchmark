#!/usr/bin/env python3
"""
Phase C53-Discovery — Age-Stratified Analysis

Split Gaussians by age group (<100, 100-500, 500-2000, >2000).
Measure signal predictive power per age group.

Output: age_stratification.json
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
AGE_GROUPS = [
    ("age_lt_100", 0, 100),
    ("age_100_500", 100, 500),
    ("age_500_2000", 500, 2000),
    ("age_gt_2000", 2000, float("inf")),
]

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


def recall_at_k(signal, utility, k_percent):
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_util = set(np.argsort(utility)[-k:])
    top_sig = set(np.argsort(signal)[-k:])
    return len(top_util & top_sig) / len(top_util) if top_util else 0.0


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

        age = data[f"{prefix}_s_age"].astype(np.float64)
        n = len(age)

        signals = {}
        for sig in SIGNALS:
            key = f"{prefix}_s_{sig}"
            if key in data:
                signals[sig] = data[key].astype(np.float64)

        # Use Δ=50 as the primary delta for age analysis
        d = 50
        d_prefix = f"{prefix}_d{d}"
        if f"{d_prefix}_alive" not in data:
            d = 10
            d_prefix = f"{prefix}_d{d}"

        if f"{d_prefix}_alive" not in data:
            continue

        alive = data[f"{d_prefix}_alive"]

        utilities = {}
        for util in UTILITIES:
            util_key = f"{d_prefix}_{util}"
            if util_key in data:
                utilities[util] = data[util_key].astype(np.float64)

        cp_results = {"iter": cp, "delta": d, "age_groups": {}}

        # Age group distribution
        for group_name, lo, hi in AGE_GROUPS:
            mask = (age >= lo) & (age < hi)
            cp_results["age_groups"][group_name] = {"n": int(mask.sum())}

        # Per age group: signal × utility correlation
        for group_name, lo, hi in AGE_GROUPS:
            age_mask = (age >= lo) & (age < hi)
            if age_mask.sum() < 50:
                continue

            group_data = {"n": int(age_mask.sum())}

            for util_name, util_vals in utilities.items():
                mask = age_mask & alive
                if mask.sum() < 50:
                    continue

                group_data[util_name] = {}
                for sig, sig_vals in signals.items():
                    if np.std(sig_vals[mask]) < 1e-12:
                        continue
                    group_data[util_name][sig] = {
                        "pearson": safe_pearson(sig_vals[mask], util_vals[mask]),
                        "spearman": safe_spearman(sig_vals[mask], util_vals[mask]),
                        "recall_10": recall_at_k(sig_vals[mask], util_vals[mask], 10),
                        "recall_20": recall_at_k(sig_vals[mask], util_vals[mask], 20),
                    }

            cp_results["age_groups"][group_name] = group_data

        results["checkpoints"][str(cp)] = cp_results

    # Summary: best signal per age group
    summary = {"best_signal_by_age": {}}
    for group_name, _, _ in AGE_GROUPS:
        summary["best_signal_by_age"][group_name] = {}
        for util in UTILITIES:
            best_sig = None
            best_pearson = -2
            for sig in SIGNALS:
                pearsons = []
                for cp_key, cp_data in results["checkpoints"].items():
                    if group_name in cp_data["age_groups"]:
                        g = cp_data["age_groups"][group_name]
                        if util in g and sig in g[util]:
                            pearsons.append(g[util][sig]["pearson"])
                if pearsons:
                    mean_p = float(np.mean(pearsons))
                    if mean_p > best_pearson:
                        best_pearson = mean_p
                        best_sig = sig
                    summary["best_signal_by_age"][group_name][util] = {
                        "best_signal": best_sig,
                        "best_pearson": best_pearson,
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

    out_file = RESULT_DIR / "age_stratification.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
