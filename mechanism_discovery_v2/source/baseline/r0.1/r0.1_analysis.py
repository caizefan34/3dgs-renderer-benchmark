"""
R0.1 Analysis — Process continuation outputs into final evidence.

Aggregates data from 4 continuation windows (2000, 5000, 10000, 14000) and produces:
  - c49_gopt_concentration.json: G_opt concentration per iteration per window
  - c49_gdens_concentration.json: G_dens concentration per iteration per window
  - c50_true_lag1_gopt.json: True lag-1 G_opt metrics (distributions over all pairs)
  - c50_true_lag1_gdens.json: True lag-1 G_dens metrics
  - c50_topology_stratified.json: C50 stratified by topology event type
  - c50_camera_conditioned.json: C50 stratified by camera similarity
  - c53_true_lag1_workload.json: True lag-1 workload persistence
  - c53_camera_conditioned.json: C53 stratified by camera similarity
  - final_go_no_go.json: C51 Go/No-Go decision

Usage: python3 r0.1_analysis.py
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Tuple


# === Configuration ===
WINDOWS = ["2000", "5000", "10000", "14000"]
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "results", "reference_v1", "r0.1")


def load_window(window: str) -> dict:
    """Load all JSON outputs from a continuation window."""
    wdir = os.path.join(BASE_DIR, f"window_{window}")
    data = {}
    for fname in os.listdir(wdir):
        if fname.endswith(".json"):
            with open(os.path.join(wdir, fname)) as f:
                key = fname.replace(".json", "")
                data[key] = json.load(f)
    return data


def aggregate_concentration(all_windows: dict, signal_key: str) -> dict:
    """Aggregate C49 concentration data across windows.

    Returns per-window, per-iteration concentration stats.
    Also computes summary statistics (mean across iterations in mature windows).
    """
    result = {}
    for window in WINDOWS:
        if window not in all_windows:
            continue
        wdata = all_windows[window]
        if signal_key not in wdata:
            continue
        result[window] = wdata[signal_key]

    # Compute summary for mature windows (5000, 10000, 14000)
    mature_windows = ["5000", "10000", "14000"]
    mature_stats = {}
    for metric in ["gini", "top1_mass", "top5_mass", "top10_mass", "top20_mass",
                    "top32_mass", "top50_mass"]:
        vals = []
        for w in mature_windows:
            if w in result:
                for iter_str, stats in result[w].items():
                    if metric in stats:
                        vals.append(stats[metric])
        if vals:
            mature_stats[metric] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }

    return {"per_window": result, "mature_summary": mature_stats}


def aggregate_lag1(all_windows: dict, signal_key: str) -> dict:
    """Aggregate lag-1 metrics across all iteration pairs in all windows.

    Returns distributions (mean, median, p10, p90) over all pairs.
    """
    all_metrics = {}  # metric_name -> list of values
    per_window = {}

    for window in WINDOWS:
        if window not in all_windows:
            continue
        wdata = all_windows[window]
        if signal_key not in wdata:
            continue

        window_metrics = {}
        lag1_data = wdata[signal_key]

        for metric_name in ["pearson", "spearman", "top1_jaccard", "top5_jaccard",
                             "top10_jaccard", "top20_jaccard", "top32_jaccard",
                             "top50_jaccard", "recall10", "recall20", "n_matched"]:
            vals = []
            for iter_str, metrics in lag1_data.items():
                if metric_name in metrics:
                    vals.append(metrics[metric_name])
            if vals:
                window_metrics[metric_name] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n": len(vals),
                }
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].extend(vals)

        per_window[window] = window_metrics

    # Aggregate distributions
    overall = {}
    for metric_name, vals in all_metrics.items():
        overall[metric_name] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals),
        }

    return {"per_window": per_window, "overall": overall}


def stratify_by_topology(all_windows: dict, signal_key: str) -> dict:
    """Stratify lag-1 metrics by topology event type.

    Categories:
      NO_TOPOLOGY_EVENT: no densification or opacity reset between t and t+1
      AFTER_CLONE_SPLIT_EVENT: densification happened at t (iteration t is a densification iter)
      AFTER_PRUNE_RESET_EVENT: opacity reset happened at t (but no densification)
    """
    categories = {
        "NO_TOPOLOGY_EVENT": {},
        "AFTER_CLONE_SPLIT_EVENT": {},
        "AFTER_PRUNE_RESET_EVENT": {},
    }

    for window in WINDOWS:
        if window not in all_windows:
            continue
        wdata = all_windows[window]
        if signal_key not in wdata:
            continue
        if "topology_events" not in wdata:
            continue

        # Build set of densification iterations
        densify_iters = set()
        for event in wdata["topology_events"].get("events", []):
            densify_iters.add(event["iteration"])

        lag1_data = wdata[signal_key]

        for iter_str, metrics in lag1_data.items():
            iter_num = int(iter_str)
            # The lag-1 pair is (iter_num-1, iter_num). The topology event
            # happens at iter_num-1 (the previous iteration).
            prev_iter = iter_num - 1

            if prev_iter in densify_iters:
                cat = "AFTER_CLONE_SPLIT_EVENT"
            elif prev_iter % 3000 == 0 and prev_iter > 0:
                cat = "AFTER_PRUNE_RESET_EVENT"
            else:
                cat = "NO_TOPOLOGY_EVENT"

            for metric_name, val in metrics.items():
                if metric_name == "n_matched":
                    continue
                if metric_name not in categories[cat]:
                    categories[cat][metric_name] = []
                categories[cat][metric_name].append(val)

    # Compute distributions
    result = {}
    for cat, metrics in categories.items():
        result[cat] = {}
        for metric_name, vals in metrics.items():
            if vals:
                result[cat][metric_name] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                    "n": len(vals),
                }

    return result


def stratify_by_camera(all_windows: dict, signal_key: str) -> dict:
    """Stratify lag-1 metrics by camera transition similarity.

    Categories based on camera-center distance:
      similar: distance < 0.5 (tercile 1)
      medium: 0.5 <= distance < 2.0 (tercile 2)
      dissimilar: distance >= 2.0 (tercile 3)

    Also stratify by view-direction angle:
      similar_angle: angle < 10 degrees
      medium_angle: 10 <= angle < 30 degrees
      dissimilar_angle: angle >= 30 degrees
    """
    # Collect all camera transitions
    all_transitions = []
    for window in WINDOWS:
        if window not in all_windows:
            continue
        wdata = all_windows[window]
        if "camera_transitions" not in wdata or signal_key not in wdata:
            continue

        cam_data = wdata["camera_transitions"]
        lag1_data = wdata[signal_key]

        for iter_str, metrics in lag1_data.items():
            if iter_str in cam_data:
                trans = cam_data[iter_str]
                all_transitions.append({
                    "window": window,
                    "iteration": int(iter_str),
                    "center_dist": trans["center_dist"],
                    "view_angle": trans["view_angle"],
                    "metrics": metrics,
                })

    if not all_transitions:
        return {"error": "no camera transition data found"}

    # Compute terciles for center distance
    center_dists = [t["center_dist"] for t in all_transitions]
    t1 = float(np.percentile(center_dists, 33.3))
    t2 = float(np.percentile(center_dists, 66.7))

    # Stratify by center distance
    categories = {
        "similar": {"threshold": f"< {t1:.3f}", "metrics": {}},
        "medium": {"threshold": f"{t1:.3f} - {t2:.3f}", "metrics": {}},
        "dissimilar": {"threshold": f">= {t2:.3f}", "metrics": {}},
    }

    for trans in all_transitions:
        if trans["center_dist"] < t1:
            cat = "similar"
        elif trans["center_dist"] < t2:
            cat = "medium"
        else:
            cat = "dissimilar"

        for metric_name, val in trans["metrics"].items():
            if metric_name == "n_matched":
                continue
            if metric_name not in categories[cat]["metrics"]:
                categories[cat]["metrics"][metric_name] = []
            categories[cat]["metrics"][metric_name].append(val)

    # Compute distributions
    result = {
        "center_dist_terciles": {"t1": t1, "t2": t2},
        "by_center_distance": {},
    }
    for cat, data in categories.items():
        result["by_center_distance"][cat] = {
            "threshold": data["threshold"],
            "metrics": {},
        }
        for metric_name, vals in data["metrics"].items():
            if vals:
                result["by_center_distance"][cat]["metrics"][metric_name] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "n": len(vals),
                }

    # Also stratify by view angle
    view_angles = [t["view_angle"] for t in all_transitions]
    a1 = 10.0  # fixed threshold
    a2 = 30.0

    angle_cats = {
        "similar_angle": {"threshold": f"< {a1}°", "metrics": {}},
        "medium_angle": {"threshold": f"{a1}° - {a2}°", "metrics": {}},
        "dissimilar_angle": {"threshold": f">= {a2}°", "metrics": {}},
    }

    for trans in all_transitions:
        if trans["view_angle"] < a1:
            cat = "similar_angle"
        elif trans["view_angle"] < a2:
            cat = "medium_angle"
        else:
            cat = "dissimilar_angle"

        for metric_name, val in trans["metrics"].items():
            if metric_name == "n_matched":
                continue
            if metric_name not in angle_cats[cat]["metrics"]:
                angle_cats[cat]["metrics"][metric_name] = []
            angle_cats[cat]["metrics"][metric_name].append(val)

    result["by_view_angle"] = {}
    for cat, data in angle_cats.items():
        result["by_view_angle"][cat] = {
            "threshold": data["threshold"],
            "metrics": {},
        }
        for metric_name, vals in data["metrics"].items():
            if vals:
                result["by_view_angle"][cat]["metrics"][metric_name] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "n": len(vals),
                }

    return result


def compute_gopt_gdens_correlation(all_windows: dict) -> dict:
    """Compute correlation between G_opt and G_dens concentration metrics.

    For each iteration, compare the Gini and top-K mass of G_opt vs G_dens.
    """
    gopt_ginis = []
    gdens_ginis = []
    gopt_top10 = []
    gdens_top10 = []

    for window in WINDOWS:
        if window not in all_windows:
            continue
        wdata = all_windows[window]
        gopt_data = wdata.get("c49_gopt_concentration", {})
        gdens_data = wdata.get("c49_gdens_concentration", {})

        for iter_str in gopt_data:
            if iter_str in gdens_data:
                gopt_ginis.append(gopt_data[iter_str].get("gini", 0))
                gdens_ginis.append(gdens_data[iter_str].get("gini", 0))
                gopt_top10.append(gopt_data[iter_str].get("top10_mass", 0))
                gdens_top10.append(gdens_data[iter_str].get("top10_mass", 0))

    result = {
        "n_iterations": len(gopt_ginis),
    }

    if len(gopt_ginis) > 5:
        gopt_arr = np.array(gopt_ginis)
        gdens_arr = np.array(gdens_ginis)
        if np.std(gopt_arr) > 1e-10 and np.std(gdens_arr) > 1e-10:
            result["gini_pearson"] = float(np.corrcoef(gopt_arr, gdens_arr)[0, 1])
        else:
            result["gini_pearson"] = 0.0

        gopt_t10 = np.array(gopt_top10)
        gdens_t10 = np.array(gdens_top10)
        if np.std(gopt_t10) > 1e-10 and np.std(gdens_t10) > 1e-10:
            result["top10_pearson"] = float(np.corrcoef(gopt_t10, gdens_t10)[0, 1])
        else:
            result["top10_pearson"] = 0.0

        result["gopt_gini_mean"] = float(np.mean(gopt_ginis))
        result["gdens_gini_mean"] = float(np.mean(gdens_ginis))
        result["gopt_top10_mean"] = float(np.mean(gopt_top10))
        result["gdens_top10_mean"] = float(np.mean(gdens_top10))

    return result


def compute_go_no_go(analysis: dict) -> dict:
    """Compute final C51 Go/No-Go decision.

    GO gate:
      Top50 G_opt mass >= 80%
      AND Top50 lag-1 recall >= 80%

    MODIFY:
      Gradient concentration exists but lag-1 predictability is weak

    DROP_MAINLINE:
      G_opt concentration weak AND/OR lag-1 predictability weak
    """
    # Extract mature G_opt concentration
    gopt_conc = analysis.get("c49_gopt_concentration", {})
    mature = gopt_conc.get("mature_summary", {})
    top50_mass = mature.get("top50_mass", {}).get("mean", 0)
    top10_mass = mature.get("top10_mass", {}).get("mean", 0)
    gini = mature.get("gini", {}).get("mean", 0)

    # Extract mature G_opt lag-1
    gopt_lag1 = analysis.get("c50_true_lag1_gopt", {})
    overall = gopt_lag1.get("overall", {})
    top50_jaccard = overall.get("top50_jaccard", {}).get("median", 0)
    recall20 = overall.get("recall20", {}).get("median", 0)
    pearson = overall.get("pearson", {}).get("median", 0)
    spearman = overall.get("spearman", {}).get("median", 0)

    # Topology stratification
    topo_strat = analysis.get("c50_topology_stratified", {})
    no_topo = topo_strat.get("NO_TOPOLOGY_EVENT", {})
    no_topo_pearson = no_topo.get("pearson", {}).get("median", 0)
    after_topo = topo_strat.get("AFTER_CLONE_SPLIT_EVENT", {})
    after_topo_pearson = after_topo.get("pearson", {}).get("median", 0)

    # Camera stratification
    cam_strat = analysis.get("c50_camera_conditioned", {})
    by_dist = cam_strat.get("by_center_distance", {})
    similar_pearson = by_dist.get("similar", {}).get("metrics", {}).get("pearson", {}).get("mean", 0)
    dissimilar_pearson = by_dist.get("dissimilar", {}).get("metrics", {}).get("pearson", {}).get("mean", 0)

    # C53 workload
    c53_lag1 = analysis.get("c53_true_lag1_workload", {})
    c53_overall = c53_lag1.get("overall", {})
    c53_pearson = c53_overall.get("pearson", {}).get("median", 0)
    c53_top10_jaccard = c53_overall.get("top10_jaccard", {}).get("median", 0)

    # Gate evaluation
    concentration_ok = top50_mass >= 0.80
    predictability_ok = top50_jaccard >= 0.80

    evidence = {
        "top50_gopt_mass": top50_mass,
        "top50_gopt_threshold": 0.80,
        "top50_lag1_jaccard_median": top50_jaccard,
        "top50_lag1_threshold": 0.80,
        "gini_gopt": gini,
        "top10_gopt_mass": top10_mass,
        "pearson_gopt_median": pearson,
        "spearman_gopt_median": spearman,
        "recall20_median": recall20,
        "no_topology_pearson": no_topo_pearson,
        "after_topology_pearson": after_topo_pearson,
        "similar_camera_pearson": similar_pearson,
        "dissimilar_camera_pearson": dissimilar_pearson,
        "c53_workload_pearson": c53_pearson,
        "c53_workload_top10_jaccard": c53_top10_jaccard,
    }

    if concentration_ok and predictability_ok:
        decision = "C51_GO"
        reason = (f"G_opt concentration sufficient (top50={top50_mass:.3f} >= 0.80) "
                  f"AND lag-1 predictability sufficient (top50_jaccard={top50_jaccard:.3f} >= 0.80)")
    elif top10_mass >= 0.50 or gini >= 0.50:
        decision = "C51_MODIFY"
        reason = (f"G_opt concentration exists (top10={top10_mass:.3f}, gini={gini:.3f}) "
                  f"but lag-1 predictability weak (top50_jaccard={top50_jaccard:.3f} < 0.80). "
                  f"Previous-gradient masking unsupported. Investigate instantaneous mechanism only.")
    else:
        decision = "C51_DROP_MAINLINE"
        reason = (f"G_opt concentration weak (top50={top50_mass:.3f}, top10={top10_mass:.3f}, "
                  f"gini={gini:.3f}) AND/OR lag-1 predictability weak "
                  f"(top50_jaccard={top50_jaccard:.3f}). "
                  f"Do not revalidate C51. Retain CUDA sparse backward as systems asset only.")

    return {
        "decision": decision,
        "reason": reason,
        "evidence": evidence,
    }


def main():
    # Load all window data
    all_windows = {}
    for window in WINDOWS:
        wdir = os.path.join(BASE_DIR, f"window_{window}")
        if os.path.exists(wdir):
            all_windows[window] = load_window(window)
            print(f"  Loaded window {window}: {len(all_windows[window])} files")
        else:
            print(f"  WARNING: window {window} not found at {wdir}")

    if not all_windows:
        print("ERROR: No window data found")
        return

    analysis = {}

    # C49: Concentration
    print("\n=== C49: Gradient concentration ===")
    analysis["c49_gopt_concentration"] = aggregate_concentration(all_windows, "c49_gopt_concentration")
    analysis["c49_gdens_concentration"] = aggregate_concentration(all_windows, "c49_gdens_concentration")

    # G_opt vs G_dens correlation
    analysis["gopt_gdens_correlation"] = compute_gopt_gdens_correlation(all_windows)
    print(f"  G_opt vs G_dens Gini Pearson: {analysis['gopt_gdens_correlation'].get('gini_pearson', 'N/A')}")

    # C50: True lag-1
    print("\n=== C50: True lag-1 temporal predictability ===")
    analysis["c50_true_lag1_gopt"] = aggregate_lag1(all_windows, "c50_true_lag1_gopt")
    analysis["c50_true_lag1_gdens"] = aggregate_lag1(all_windows, "c50_true_lag1_gdens")
    analysis["c50_topology_stratified"] = stratify_by_topology(all_windows, "c50_true_lag1_gopt")
    analysis["c50_camera_conditioned"] = stratify_by_camera(all_windows, "c50_true_lag1_gopt")

    # C53: True lag-1 workload
    print("\n=== C53: True lag-1 workload persistence ===")
    analysis["c53_true_lag1_workload"] = aggregate_lag1(all_windows, "c53_true_lag1_workload")
    analysis["c53_camera_conditioned"] = stratify_by_camera(all_windows, "c53_true_lag1_workload")

    # C51 Go/No-Go
    print("\n=== C51 Go/No-Go ===")
    decision = compute_go_no_go(analysis)
    analysis["final_go_no_go"] = decision
    print(f"  Decision: {decision['decision']}")
    print(f"  Reason: {decision['reason']}")

    # Save all outputs
    out_dir = BASE_DIR
    for fname, data in [
        ("c49_gopt_concentration.json", analysis["c49_gopt_concentration"]),
        ("c49_gdens_concentration.json", analysis["c49_gdens_concentration"]),
        ("c50_true_lag1_gopt.json", analysis["c50_true_lag1_gopt"]),
        ("c50_true_lag1_gdens.json", analysis["c50_true_lag1_gdens"]),
        ("c50_topology_stratified.json", analysis["c50_topology_stratified"]),
        ("c50_camera_conditioned.json", analysis["c50_camera_conditioned"]),
        ("c53_true_lag1_workload.json", analysis["c53_true_lag1_workload"]),
        ("c53_camera_conditioned.json", analysis["c53_camera_conditioned"]),
        ("final_go_no_go.json", analysis["final_go_no_go"]),
    ]:
        # Also save correlation
        pass

    # Save correlation separately
    with open(os.path.join(out_dir, "gopt_gdens_correlation.json"), "w") as f:
        json.dump(analysis["gopt_gdens_correlation"], f, indent=2)

    for fname, data in [
        ("c49_gopt_concentration.json", analysis["c49_gopt_concentration"]),
        ("c49_gdens_concentration.json", analysis["c49_gdens_concentration"]),
        ("c50_true_lag1_gopt.json", analysis["c50_true_lag1_gopt"]),
        ("c50_true_lag1_gdens.json", analysis["c50_true_lag1_gdens"]),
        ("c50_topology_stratified.json", analysis["c50_topology_stratified"]),
        ("c50_camera_conditioned.json", analysis["c50_camera_conditioned"]),
        ("c53_true_lag1_workload.json", analysis["c53_true_lag1_workload"]),
        ("c53_camera_conditioned.json", analysis["c53_camera_conditioned"]),
        ("final_go_no_go.json", analysis["final_go_no_go"]),
    ]:
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {fname}")

    print(f"\n=== Analysis complete. Outputs in {out_dir} ===")


if __name__ == "__main__":
    main()
