#!/usr/bin/env python3
"""
Phase C53-Discovery — Final Comparison & Signal Ranking

Cross-scene comparison, final signal ranking, and all 4 output tables.

Outputs: cross_scene_comparison.json, final_signal_ranking.json
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

# Utility targets for the main comparison
MAIN_UTILITIES = ["grad_sum", "update_xyz", "vis_count"]


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def recall_at_k(signal, utility, k_percent):
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_util = set(np.argsort(utility)[-k:])
    top_sig = set(np.argsort(signal)[-k:])
    return len(top_util & top_sig) / len(top_util) if top_util else 0.0


def coverage_at_k(signal, utility, k_percent):
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_sig = np.argsort(signal)[-k:]
    total = utility.sum()
    if total < 1e-12:
        return 0.0
    return float(utility[top_sig].sum() / total)


def compute_scene_signal_stats(scene, suffix=""):
    """Compute per-signal statistics across all checkpoints and deltas."""
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        return None

    data = np.load(filepath, allow_pickle=True)
    stats = {}

    for sig in SIGNALS:
        sig_stats = {"pearson_by_util": {}, "recall_by_util": {}, "coverage_by_util": {}}

        for util in MAIN_UTILITIES:
            for d in DELTAS:
                pearsons = []
                recalls = {10: [], 20: [], 50: []}
                coverages = {10: [], 20: [], 50: []}

                for cp in CHECKPOINTS:
                    prefix = f"cp{cp}"
                    if f"{prefix}_s_{sig}" not in data:
                        continue
                    d_prefix = f"{prefix}_d{d}"
                    if f"{d_prefix}_{util}" not in data:
                        continue
                    if f"{d_prefix}_alive" not in data:
                        continue

                    sig_vals = data[f"{prefix}_s_{sig}"].astype(float)
                    util_vals = data[f"{d_prefix}_{util}"].astype(float)
                    alive = data[f"{d_prefix}_alive"]
                    mask = alive.copy()

                    if mask.sum() < 50 or np.std(sig_vals[mask]) < 1e-12:
                        continue

                    pearsons.append(safe_pearson(sig_vals[mask], util_vals[mask]))
                    for k in [10, 20, 50]:
                        recalls[k].append(recall_at_k(sig_vals[mask], util_vals[mask], k))
                        coverages[k].append(coverage_at_k(sig_vals[mask], util_vals[mask], k))

                if pearsons:
                    key = f"{util}_d{d}"
                    sig_stats["pearson_by_util"][key] = {
                        "mean": float(np.mean(pearsons)),
                        "std": float(np.std(pearsons)),
                        "n": len(pearsons),
                    }
                    sig_stats["recall_by_util"][key] = {
                        f"recall_{k}_mean": float(np.mean(recalls[k])) for k in [10, 20, 50] if recalls[k]
                    }
                    sig_stats["coverage_by_util"][key] = {
                        f"coverage_{k}_mean": float(np.mean(coverages[k])) for k in [10, 20, 50] if coverages[k]
                    }

        stats[sig] = sig_stats

    return stats


def rank_signals(scene_stats):
    """Rank signals by composite score across utility targets and deltas."""
    rankings = {}

    for sig in SIGNALS:
        scores = {"pearson_grad": [], "pearson_update": [], "pearson_vis": [],
                   "recall10_grad": [], "recall10_update": [], "recall10_vis": [],
                   "coverage20_grad": [], "coverage20_update": [], "coverage20_vis": []}

        for scene, stats in scene_stats.items():
            if stats is None or sig not in stats:
                continue
            for key, val in stats[sig]["pearson_by_util"].items():
                if "grad_sum" in key:
                    scores["pearson_grad"].append(val["mean"])
                elif "update_xyz" in key:
                    scores["pearson_update"].append(val["mean"])
                elif "vis_count" in key:
                    scores["pearson_vis"].append(val["mean"])

            for key, val in stats[sig]["recall_by_util"].items():
                if "grad_sum" in key and "recall_10_mean" in val:
                    scores["recall10_grad"].append(val["recall_10_mean"])
                elif "update_xyz" in key and "recall_10_mean" in val:
                    scores["recall10_update"].append(val["recall_10_mean"])
                elif "vis_count" in key and "recall_10_mean" in val:
                    scores["recall10_vis"].append(val["recall_10_mean"])

        rankings[sig] = {
            "mean_pearson_grad": float(np.mean(scores["pearson_grad"])) if scores["pearson_grad"] else 0,
            "mean_pearson_update": float(np.mean(scores["pearson_update"])) if scores["pearson_update"] else 0,
            "mean_pearson_vis": float(np.mean(scores["pearson_vis"])) if scores["pearson_vis"] else 0,
            "mean_recall10_grad": float(np.mean(scores["recall10_grad"])) if scores["recall10_grad"] else 0,
            "mean_recall10_update": float(np.mean(scores["recall10_update"])) if scores["recall10_update"] else 0,
            "mean_recall10_vis": float(np.mean(scores["recall10_vis"])) if scores["recall10_vis"] else 0,
            "n_scenes_grad": len(scores["pearson_grad"]),
            "n_scenes_update": len(scores["pearson_update"]),
            "n_scenes_vis": len(scores["pearson_vis"]),
        }

    return rankings


def assign_rating(rankings):
    """Assign STRONG/MODERATE/WEAK/DROP rating to each signal."""
    ratings = {}

    for sig in SIGNALS:
        r = rankings[sig]
        # Primary criterion: mean Pearson across utility targets and scenes
        mean_pearson = np.mean([r["mean_pearson_grad"], r["mean_pearson_update"], r["mean_pearson_vis"]])
        max_pearson = max(r["mean_pearson_grad"], r["mean_pearson_update"], r["mean_pearson_vis"])

        # Cross-scene consistency: how many scenes have non-zero correlation
        n_scenes = max(r["n_scenes_grad"], r["n_scenes_update"], r["n_scenes_vis"])

        # Temporal horizon: check if correlation persists at Δ=100
        # (this is a simplified check; detailed analysis is in temporal_horizon.json)

        if max_pearson > 0.15 and n_scenes >= 2:
            rating = "STRONG"
        elif max_pearson > 0.08 and n_scenes >= 2:
            rating = "MODERATE"
        elif max_pearson > 0.03:
            rating = "WEAK"
        else:
            rating = "DROP"

        ratings[sig] = {
            "rating": rating,
            "mean_pearson_overall": float(mean_pearson),
            "max_pearson": float(max_pearson),
            "n_scenes": n_scenes,
            "best_utility": max(
                [("grad", r["mean_pearson_grad"]), ("update", r["mean_pearson_update"]), ("vis", r["mean_pearson_vis"])],
                key=lambda x: x[1]
            )[0],
        }

    return ratings


def main():
    # Compute per-scene stats
    scene_stats = {}
    for scene in ["room", "garden", "bicycle"]:
        stats = compute_scene_signal_stats(scene)
        if stats:
            scene_stats[scene] = stats
        stats_ctrl = compute_scene_signal_stats(scene, "_seed123")
        if stats_ctrl:
            scene_stats[f"{scene}_seed123"] = stats_ctrl

    # Cross-scene comparison table (Table 3 from spec)
    cross_scene = {}
    for sig in SIGNALS:
        cross_scene[sig] = {}
        for scene in ["room", "garden", "bicycle"]:
            if scene in scene_stats and sig in scene_stats[scene]:
                s = scene_stats[scene][sig]
                # Mean Pearson across all utils and deltas
                all_p = [v["mean"] for v in s["pearson_by_util"].values()]
                cross_scene[sig][scene] = {
                    "mean_pearson": float(np.mean(all_p)) if all_p else 0,
                    "n_measurements": len(all_p),
                }

    # Consistency: std across scenes
    for sig in SIGNALS:
        pears = [cross_scene[sig][s]["mean_pearson"] for s in ["room", "garden", "bicycle"] if s in cross_scene[sig]]
        cross_scene[sig]["cross_scene_consistency"] = {
            "std_across_scenes": float(np.std(pears)) if len(pears) > 1 else 0,
            "n_scenes": len(pears),
            "all_positive": all(p > 0 for p in pears) if pears else False,
        }

    out_file = RESULT_DIR / "cross_scene_comparison.json"
    with open(out_file, 'w') as f:
        json.dump(cross_scene, f, indent=2)
    print(f"Saved {out_file}")

    # Final signal ranking
    rankings = rank_signals(scene_stats)
    ratings = assign_rating(rankings)

    final_ranking = {
        "rankings": rankings,
        "ratings": ratings,
        "rating_criteria": {
            "STRONG": "max_pearson > 0.15 and present in >=2 scenes",
            "MODERATE": "max_pearson > 0.08 and present in >=2 scenes",
            "WEAK": "max_pearson > 0.03",
            "DROP": "max_pearson <= 0.03 or not cross-scene",
        },
    }

    out_file2 = RESULT_DIR / "final_signal_ranking.json"
    with open(out_file2, 'w') as f:
        json.dump(final_ranking, f, indent=2)
    print(f"Saved {out_file2}")

    # Print summary table
    print("\n=== Final Signal Ranking ===")
    print(f"| Signal | Rating | Mean Pearson (grad) | Mean Pearson (update) | Mean Pearson (vis) | Best Utility |")
    print(f"|--------|--------|-------------------|----------------------|-------------------|-------------|")
    for sig in SIGNALS:
        r = rankings[sig]
        rating = ratings[sig]["rating"]
        print(f"| {sig} | {rating} | {r['mean_pearson_grad']:.4f} | {r['mean_pearson_update']:.4f} | "
              f"{r['mean_pearson_vis']:.4f} | {ratings[sig]['best_utility']} |")


if __name__ == "__main__":
    main()
