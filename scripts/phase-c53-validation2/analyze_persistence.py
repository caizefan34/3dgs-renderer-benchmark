#!/usr/bin/env python3
"""
Phase C53-Validation2 — Workload Persistence Analysis

Experiment A: W_current → W_future (workload persistence)
Measures Pearson, Spearman, Recall@K, Coverage@K for current tiles → future tiles.
"""
import json
import numpy as np
from pathlib import Path

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-validation2")
OUT_DIR = RESULT_DIR
CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
DELTAS = [10, 50, 100]


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def recall_at_k(signal, target, k_percent):
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_sig = set(np.argsort(signal)[-k:])
    top_tgt = set(np.argsort(target)[-k:])
    return len(top_sig & top_tgt) / len(top_tgt) if top_tgt else 0.0


def coverage_at_k(signal, target, k_percent):
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_sig = np.argsort(signal)[-k:]
    total = target.sum()
    return float(target[top_sig].sum() / total) if total > 1e-12 else 0.0


def analyze_scene(scene, suffix=""):
    fp = RESULT_DIR / f"{scene}{suffix}_workload.npz"
    if not fp.exists():
        print(f"  {fp} not found")
        return None
    data = np.load(fp, allow_pickle=True)
    results = {"scene": scene + suffix, "checkpoints": {}}

    for cp in CHECKPOINTS:
        p = f"cp{cp}"
        if f"{p}_ids" not in data:
            continue
        w_current = data[f"{p}_s_current_tiles_mean"].astype(np.float64)

        cp_res = {}
        for d in DELTAS:
            dp = f"{p}_d{d}"
            if f"{dp}_alive" not in data:
                continue
            alive = data[f"{dp}_alive"]
            w_future = data[f"{dp}_tiles_mean"].astype(np.float64)

            mask = alive & (w_current > 0)  # Only visible Gaussians have meaningful current workload
            if mask.sum() < 100:
                continue

            wc = w_current[mask]
            wf = w_future[mask]

            cp_res[str(d)] = {
                "pearson": safe_pearson(wc, wf),
                "spearman": safe_spearman(wc, wf),
                "recall_10": recall_at_k(wc, wf, 10),
                "recall_20": recall_at_k(wc, wf, 20),
                "recall_50": recall_at_k(wc, wf, 50),
                "coverage_20": coverage_at_k(wc, wf, 20),
                "coverage_50": coverage_at_k(wc, wf, 50),
                "n": int(mask.sum()),
            }
        results["checkpoints"][str(cp)] = cp_res
    return results


def main():
    print("=" * 100)
    print("Phase C53-Validation2 — Workload Persistence (W_current → W_future)")
    print("=" * 100)

    scenes = ["room", "garden", "bicycle", "room_seed123", "garden_seed123"]
    all_results = {}
    for scene in scenes:
        parts = scene.split("_seed")
        base, suf = parts[0], f"_seed{parts[1]}" if len(parts) > 1 else ""
        r = analyze_scene(base, suf)
        if r:
            all_results[scene] = r

    # Cross-scene summary
    primary = ["room", "garden", "bicycle"]
    for d in DELTAS:
        print(f"\nΔ={d} (mean across room/garden/bicycle, 8 checkpoints):")
        print(f"  {'Scene':<15} {'Pearson':>10} {'Spearman':>10} {'Recall@10':>10} {'Recall@20':>10} {'Recall@50':>10} {'Cov@20':>10}")
        for scene in primary:
            vals = []
            if scene in all_results:
                for cp in CHECKPOINTS:
                    cp_str = str(cp)
                    if cp_str in all_results[scene]["checkpoints"] and str(d) in all_results[scene]["checkpoints"][cp_str]:
                        vals.append(all_results[scene]["checkpoints"][cp_str][str(d)])
            if vals:
                m = {k: np.mean([v[k] for v in vals]) for k in ["pearson", "spearman", "recall_10", "recall_20", "recall_50", "coverage_20"]}
                print(f"  {scene:<15} {m['pearson']:>10.3f} {m['spearman']:>10.3f} {m['recall_10']:>10.3f} {m['recall_20']:>10.3f} {m['recall_50']:>10.3f} {m['coverage_20']:>10.3f}")

        # Pooled mean
        all_vals = []
        for scene in primary:
            if scene in all_results:
                for cp in CHECKPOINTS:
                    cp_str = str(cp)
                    if cp_str in all_results[scene]["checkpoints"] and str(d) in all_results[scene]["checkpoints"][cp_str]:
                        all_vals.append(all_results[scene]["checkpoints"][cp_str][str(d)])
        if all_vals:
            m = {k: np.mean([v[k] for v in all_vals]) for k in ["pearson", "spearman", "recall_10", "recall_20", "recall_50", "coverage_20"]}
            s = {k: np.std([v[k] for v in all_vals]) for k in ["pearson"]}
            print(f"  {'MEAN':<15} {m['pearson']:>10.3f} {m['spearman']:>10.3f} {m['recall_10']:>10.3f} {m['recall_20']:>10.3f} {m['recall_50']:>10.3f} {m['coverage_20']:>10.3f}")
            print(f"  {'STD':<15} {s['pearson']:>10.3f}")

    out_file = OUT_DIR / "workload_persistence.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
