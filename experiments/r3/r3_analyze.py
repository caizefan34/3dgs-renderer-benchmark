#!/usr/bin/env python3
"""
R3 Aggregation and Analysis (Corrected)

Aggregates per-window JSON outputs from r3_certificate_runner.py.
Handles the corrected output format:
  results/reference_v1/r3/{5000,10000,15000}/{certificate_correctness,...}.json

Computes pooled statistics, violation reports, per-window summaries.
"""

import os, sys, json, glob, argparse
import numpy as np


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_window_dirs(top_dir):
    """Find all subdirectories that contain R3 output JSONs."""
    windows = {}
    for name in sorted(os.listdir(top_dir)):
        sub = os.path.join(top_dir, name)
        if os.path.isdir(sub) and not name.startswith('.'):
            # Check if it contains at least one R3 JSON
            jsons = [f for f in os.listdir(sub) if f.endswith('.json') and f.startswith('certificate_')]
            if jsons:
                windows[name] = sub
    return windows


def aggregate_correctness(input_dir):
    """Aggregate per-iteration correctness results from all windows."""
    results = {}
    windows = find_window_dirs(input_dir)
    
    for win_name, win_dir in windows.items():
        data = load_json(os.path.join(win_dir, "certificate_correctness.json"))
        if data and "correctness" in data:
            for iter_key, fam_data in data["correctness"].items():
                results[f"{win_name}/{iter_key}"] = fam_data
    
    return results


def aggregate_tightness(input_dir):
    """Aggregate tightness ratios from all windows."""
    results = {}
    windows = find_window_dirs(input_dir)
    
    for win_name, win_dir in windows.items():
        data = load_json(os.path.join(win_dir, "certificate_tightness.json"))
        if data and "tightness" in data:
            for iter_key, fam_data in data["tightness"].items():
                results[f"{win_name}/{iter_key}"] = fam_data
    
    return results


def compute_pooled_medians(tightness_data):
    """Compute pooled median tightness ratios across all iterations."""
    pooled = {}
    for path_key, fam_data in tightness_data.items():
        for fam, stats in fam_data.items():
            if stats.get("available", False):
                med = stats.get("median")
                if med is not None:
                    pooled.setdefault(fam, []).append(med)
    
    return {
        fam: {
            "pooled_median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_measurements": len(vals),
        }
        for fam, vals in sorted(pooled.items())
    }


def aggregate_disabled(input_dir):
    """Aggregate SPD-disabled counts from all windows."""
    disabled_info = {}
    windows = find_window_dirs(input_dir)
    
    for win_name, win_dir in windows.items():
        data = load_json(os.path.join(win_dir, "certificate_disabled.json"))
        if data and "disabled" in data:
            for iter_key, d_info in data["disabled"].items():
                disabled_info[f"{win_name}/{iter_key}"] = d_info
    
    return disabled_info


def aggregate_joint_skip(input_dir):
    """Aggregate JOINT skip-set analysis from all windows."""
    joint_info = {}
    windows = find_window_dirs(input_dir)
    
    for win_name, win_dir in windows.items():
        data = load_json(os.path.join(win_dir, "tile_gaussian_certificate.json"))
        if data and "joint_skip" in data:
            for iter_key, eps_data in data["joint_skip"].items():
                joint_info[f"{win_name}/{iter_key}"] = eps_data
    
    return joint_info


def aggregate_exact_zero(input_dir):
    """Aggregate exact-zero statistics."""
    ez_info = {}
    windows = find_window_dirs(input_dir)
    
    for win_name, win_dir in windows.items():
        data = load_json(os.path.join(win_dir, "exact_zero_statistics.json"))
        if data and "exact_zero" in data:
            for iter_key, d_info in data["exact_zero"].items():
                ez_info[f"{win_name}/{iter_key}"] = d_info
    
    return ez_info


def main():
    parser = argparse.ArgumentParser(description="R3 aggregation tool")
    parser.add_argument("--input-dir", required=True, help="Top-level R3 output dir")
    parser.add_argument("--output", required=False, help="Output summary JSON")
    args = parser.parse_args()
    
    correctness = aggregate_correctness(args.input_dir)
    tightness = aggregate_tightness(args.input_dir)
    disabled = aggregate_disabled(args.input_dir)
    exact_zero = aggregate_exact_zero(args.input_dir)
    joint_skip = aggregate_joint_skip(args.input_dir)
    
    pooled = compute_pooled_medians(tightness)
    
    # Count total violations
    total_violations = 0
    violation_by_family = {}
    for path_key, fam_data in correctness.items():
        for fam, stats in fam_data.items():
            if stats.get("available", False):
                v = stats.get("violation_count", 0)
                total_violations += v
                violation_by_family.setdefault(fam, 0)
                violation_by_family[fam] += v
    
    # Aggregate JOINT skip statistics
    joint_summary = {}
    if joint_skip:
        # Collect all epsilon keys
        all_eps = set()
        for iter_key, eps_data in joint_skip.items():
            for eps_key in eps_data.keys():
                all_eps.add(eps_key)
        
        for eps_key in sorted(all_eps):
            pair_fracs = []
            work_fracs = []
            loss_cond_fracs = []
            for iter_key, eps_data in joint_skip.items():
                if eps_key in eps_data:
                    pair_fracs.append(eps_data[eps_key].get("JOINT_SKIP_PAIR_FRACTION", 0))
                    work_fracs.append(eps_data[eps_key].get("JOINT_SKIP_WEIGHTED_WORK_FRACTION", 0))
                    loss_cond_fracs.append(eps_data[eps_key].get("LOSS_CONDITIONED_NONZERO_SKIP", 0) if "LOSS_CONDITIONED_NONZERO_SKIP" in eps_data[eps_key] else eps_data[eps_key].get("LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING", 0))
            joint_summary[eps_key] = {
                "JOINT_SKIP_PAIR_FRACTION_MEAN": float(np.mean(pair_fracs)) if pair_fracs else 0.0,
                "JOINT_SKIP_WEIGHTED_FRACTION_MEAN": float(np.mean(work_fracs)) if work_fracs else 0.0,
                "LOSS_CONDITIONED_FRACTION_MEAN": float(np.mean(loss_cond_fracs)) if loss_cond_fracs else 0.0,
            }
    
    # Per-window breakdown
    windows_seen = sorted(set(p.split("/")[0] for p in correctness.keys()))
    
    result = {
        "input_dir": args.input_dir,
        "windows_covered": windows_seen,
        "n_iterations_total": len(correctness),
        "pooled_median_tightness": pooled,
        "total_violations": total_violations,
        "violations_by_family": violation_by_family,
        "zero_violations": total_violations == 0,
        "correctness_summary": {
            "pass": total_violations == 0,
            "total_violations": total_violations,
            "violations_by_family": violation_by_family,
            "n_iterations": len(correctness),
        },
        "tightness_summary": tightness,
        "exact_zero_summary": exact_zero,
        "disabled_summary": disabled,
        "joint_skip_summary": joint_summary,
    }
    
    # Save if output specified
    if args.output:
        if os.path.isdir(args.output):
            outpath = os.path.join(args.output, "aggregated_summary.json")
        else:
            outpath = args.output
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Aggregated summary saved: {outpath}")
    
    # Print summary
    print("\n=== R3 Aggregation Summary ===")
    print(f"Windows: {windows_seen}")
    print(f"Iterations: {len(correctness)}")
    print(f"Total violations: {total_violations}")
    print(f"Zero violations: {result['correctness_summary']['pass']}")
    print("\nPooled Median Tightness (B/||g||):")
    for fam, stats in sorted(pooled.items()):
        print(f"  {fam}: med={stats['pooled_median']:.4f} mean={stats['mean']:.4f} "
              f"[{stats['min']:.4f}, {stats['max']:.4f}] n={stats['n_measurements']}")

    if joint_summary:
        print("\nJOINT Skip-Set Summary (all 4 families simultaneously):")
        for eps_key, stats in sorted(joint_summary.items()):
            print(
                f"  {eps_key}: pair={stats['JOINT_SKIP_PAIR_FRACTION_MEAN']*100:.2f}% "
                f"work={stats['JOINT_SKIP_WEIGHTED_FRACTION_MEAN']*100:.2f}% "
                f"loss-cond={stats['LOSS_CONDITIONED_FRACTION_MEAN']*100:.2f}%"
            )
    else:
        print("\nJOINT Skip-Set Summary: no data available")


if __name__ == "__main__":
    main()
