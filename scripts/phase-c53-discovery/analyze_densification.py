#!/usr/bin/env python3
"""
Phase C53-Discovery — Densification-Aware Subanalysis

Split Gaussians by future outcome (cloned, split, unchanged, pruned).
Test which signal best predicts each outcome.

Output: densification_analysis.json
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
    results = {"scene": scene, "checkpoints": {}}

    for cp in CHECKPOINTS:
        prefix = f"cp{cp}"
        if f"{prefix}_ids" not in data:
            continue

        outcome = data[f"{prefix}_outcome"]
        n = len(outcome)

        # Get signals
        signals = {}
        for sig in SIGNALS:
            key = f"{prefix}_s_{sig}"
            if key in data:
                signals[sig] = data[key].astype(np.float64)

        # Get future utility (use Δ=100 for the most complete picture)
        d = 100
        d_prefix = f"{prefix}_d{d}"
        if f"{d_prefix}_alive" not in data:
            d = 50
            d_prefix = f"{prefix}_d{d}"
        if f"{d_prefix}_alive" not in data:
            d = 10
            d_prefix = f"{prefix}_d{d}"

        alive = data[f"{d_prefix}_alive"] if f"{d_prefix}_alive" in data else np.ones(n, bool)
        grad_sum = data[f"{d_prefix}_grad_sum"] if f"{d_prefix}_grad_sum" in data else np.zeros(n)
        update_xyz = data[f"{d_prefix}_update_xyz"] if f"{d_prefix}_update_xyz" in data else np.zeros(n)

        cp_results = {"iter": cp, "delta": d, "outcome_counts": {
            "unchanged": int((outcome == 0).sum()),
            "cloned": int((outcome == 1).sum()),
            "split": int((outcome == 2).sum()),
            "pruned": int((outcome == 3).sum()),
        }}

        # Signal statistics per outcome group
        cp_results["signal_by_outcome"] = {}
        for outcome_val, outcome_name in [(0, "unchanged"), (1, "cloned"), (2, "split"), (3, "pruned")]:
            mask = outcome == outcome_val
            if mask.sum() < 10:
                cp_results["signal_by_outcome"][outcome_name] = {"n": int(mask.sum())}
                continue
            cp_results["signal_by_outcome"][outcome_name] = {"n": int(mask.sum())}
            for sig, vals in signals.items():
                cp_results["signal_by_outcome"][outcome_name][sig] = {
                    "mean": float(vals[mask].mean()),
                    "std": float(vals[mask].std()),
                }

        # Can signals predict clone vs unchanged? (binary classification)
        # Use signal as score, compute AUC-like metric (Recall@K where K = clone fraction)
        cp_results["clone_prediction"] = {}
        clone_mask = (outcome == 1)
        n_clones = int(clone_mask.sum())
        if n_clones > 10:
            # For each signal, what fraction of top-K signal Gaussians are clones?
            clone_fraction = n_clones / n
            k = max(1, int(n * clone_fraction))
            for sig, vals in signals.items():
                top_sig = set(np.argsort(vals)[-k:])
                n_clones_in_top = int(clone_mask[list(top_sig)].sum())
                recall = n_clones_in_top / n_clones if n_clones > 0 else 0
                cp_results["clone_prediction"][sig] = {
                    "clone_fraction": float(clone_fraction),
                    "recall_at_clone_fraction": float(recall),
                    "random_baseline": float(clone_fraction),
                    "lift": float(recall / clone_fraction) if clone_fraction > 0 else 1.0,
                }

        # Split prediction
        cp_results["split_prediction"] = {}
        split_mask = (outcome == 2)
        n_splits = int(split_mask.sum())
        if n_splits > 10:
            split_fraction = n_splits / n
            k = max(1, int(n * split_fraction))
            for sig, vals in signals.items():
                top_sig = set(np.argsort(vals)[-k:])
                n_splits_in_top = int(split_mask[list(top_sig)].sum())
                recall = n_splits_in_top / n_splits if n_splits > 0 else 0
                cp_results["split_prediction"][sig] = {
                    "split_fraction": float(split_fraction),
                    "recall_at_split_fraction": float(recall),
                    "random_baseline": float(split_fraction),
                    "lift": float(recall / split_fraction) if split_fraction > 0 else 1.0,
                }

        # Prune prediction
        cp_results["prune_prediction"] = {}
        prune_mask = (outcome == 3)
        n_prunes = int(prune_mask.sum())
        if n_prunes > 10:
            prune_fraction = n_prunes / n
            k = max(1, int(n * prune_fraction))
            for sig, vals in signals.items():
                # For prune, LOW signal might predict pruning (e.g., low opacity)
                bottom_sig = set(np.argsort(vals)[:k])
                n_prunes_in_bottom = int(prune_mask[list(bottom_sig)].sum())
                recall = n_prunes_in_bottom / n_prunes if n_prunes > 0 else 0
                cp_results["prune_prediction"][sig] = {
                    "prune_fraction": float(prune_fraction),
                    "recall_at_prune_fraction_bottom": float(recall),
                    "random_baseline": float(prune_fraction),
                    "lift": float(recall / prune_fraction) if prune_fraction > 0 else 1.0,
                }

        # Correlation: signal vs binary clone/split/prune indicator
        cp_results["outcome_correlation"] = {}
        for outcome_val, outcome_name in [(1, "cloned"), (2, "split"), (3, "pruned")]:
            indicator = (outcome == outcome_val).astype(float)
            if indicator.sum() < 10 or indicator.sum() > n - 10:
                continue
            cp_results["outcome_correlation"][outcome_name] = {}
            for sig, vals in signals.items():
                cp_results["outcome_correlation"][outcome_name][sig] = {
                    "pearson": safe_pearson(vals, indicator),
                }

        results["checkpoints"][str(cp)] = cp_results

    # Summary: mean lift across checkpoints
    summary = {"clone_lift": {}, "split_lift": {}, "prune_lift": {}}
    for outcome_name, pred_key, lift_key in [
        ("cloned", "clone_prediction", "clone_lift"),
        ("split", "split_prediction", "split_lift"),
        ("pruned", "prune_prediction", "prune_lift"),
    ]:
        for sig in SIGNALS:
            lifts = []
            for cp_key, cp_data in results["checkpoints"].items():
                if pred_key in cp_data and sig in cp_data[pred_key]:
                    lifts.append(cp_data[pred_key][sig]["lift"])
            if lifts:
                summary[lift_key][sig] = {
                    "mean_lift": float(np.mean(lifts)),
                    "std_lift": float(np.std(lifts)),
                    "n_checkpoints": len(lifts),
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

    out_file = RESULT_DIR / "densification_analysis.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
