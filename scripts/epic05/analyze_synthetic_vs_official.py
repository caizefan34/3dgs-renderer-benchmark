#!/usr/bin/env python3
"""
Synthetic vs Official Dataset Comparative Analysis.

Combines synthetic workload scaling results (50K, 200K, 400K Gaussians)
with official real-scene results to answer:

    Q10. Does synthetic scaling trend replicate on real scenes?
    Q11. Is Gaussian count sufficient to explain real-scene speedup?
    Q12. Are there screen-space / visibility / density factors at play?

Usage:
    python scripts/epic05/analyze_synthetic_vs_official.py
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Expected result locations
# ---------------------------------------------------------------------------
SYNTHETIC_RESULTS_PATHS = [
    REPO_ROOT / "results" / "epic05" / "aggregated",
    REPO_ROOT / "data" / "results",
]

OFFICIAL_AGGREGATED_PATH = (
    REPO_ROOT / "results" / "epic05" / "official" / "aggregated" / "official_aggregated.json"
)

# ---------------------------------------------------------------------------
# Synthetic scene reference
# ---------------------------------------------------------------------------
SYNTHETIC_SCENES = {
    "50k": {"label": "50K synthetic", "gaussians": 50000},
    "200k": {"label": "200K synthetic", "gaussians": 200000},
    "400k": {"label": "400K synthetic", "gaussians": 400000},
}


def load_synthetic_results() -> Dict[str, Dict]:
    """Load synthetic tile-size speed results from the existing epic05 aggregates.

    This scans the aggregated results directory for synthetic scene data
    containing tile_size comparisons (tile16 mean_ms, tile32 mean_ms).
    """
    results = {}

    # Try structured aggregation files
    for agg_dir in [REPO_ROOT / "results" / "epic05" / "aggregated"]:
        if not agg_dir.exists():
            continue
        for fpath in sorted(agg_dir.glob("*aggregat*")):
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            # Check if this contains synthetic results with tile_size data
            for scene_key in ("50k", "200k", "400k", "scene_50k", "scene_200k", "scene_400k"):
                if scene_key in data:
                    entry = data[scene_key]
                    if isinstance(entry, dict):
                        results[scene_key] = entry

    # Try raw results
    raw_dir = REPO_ROOT / "results" / "epic05" / "raw"
    if raw_dir.exists():
        for fpath in sorted(raw_dir.glob("*.json")):
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            # Look for tile_size benchmark data
            scenes = data.get("scenes", data)
            for sk, sv in scenes.items():
                sk_lower = sk.lower()
                if any(
                    s in sk_lower for s in ["50k", "200k", "400k", "scene_50"]
                ):
                    results[sk] = sv if isinstance(sv, dict) else {}

    return results


def load_official_aggregated(path: Path) -> List[Dict]:
    """Load the official scene aggregated results."""
    if not path.exists():
        print(f"WARNING: Official aggregated results not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rows", [])


def load_synthetic_speed_results_from_optimization() -> Dict[str, Dict]:
    """Alternative: extract tile size speed data from optimization matrix runs.

    Scans raw result files for tile16/tile32 mean_ms.
    """
    synthetic = {}
    raw_dir = REPO_ROOT / "results" / "epic05" / "raw"
    if not raw_dir.exists():
        return synthetic

    for fpath in sorted(raw_dir.glob("*optimization*")):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        scenarios = data.get("results", data.get("scenarios", data.get("scenes", data)))
        if not isinstance(scenarios, dict):
            continue

        for scene_key, scene_data in scenarios.items():
            if not isinstance(scene_data, dict):
                continue
            # Check for tile16/tile32 keys
            tile16 = scene_data.get("tile16", scene_data.get("M1b"))
            tile32 = scene_data.get("tile32", scene_data.get("M1c"))

            if tile16 is None and "results" in scene_data:
                # Nested results
                tile16 = scene_data["results"].get("tile16", scene_data["results"].get("M1b"))
                tile32 = scene_data["results"].get("tile32", scene_data["results"].get("M1c"))

            if tile16 and tile32:
                t16_ms = tile16.get("mean_ms", tile16.get("mean_latency_ms", tile16.get("latency_ms", None)))
                t32_ms = tile32.get("mean_ms", tile32.get("mean_latency_ms", tile32.get("latency_ms", None)))

                if t16_ms and t32_ms:
                    num_g = scene_data.get("num_gaussians", scene_data.get("gaussians", 0))
                    if isinstance(num_g, str):
                        num_g = int(num_g.replace("K", "000").replace("k", "000"))
                    synthetic[scene_key] = {
                        "num_gaussians": int(num_g) if num_g else 0,
                        "tile16_mean_ms": t16_ms,
                        "tile32_mean_ms": t32_ms,
                        "speedup": t16_ms / t32_ms if t32_ms > 0 else None,
                    }
                    break

    return synthetic


def find_tile_data(entry: Dict, tile_key: str) -> Optional[Dict]:
    """Extract tile size data from various key schemas."""
    candidates = [tile_key, f"tile_size_{tile_key.replace('tile', '')}",
                  tile_key.replace("tile", "M1")[0] + tile_key.replace("tile", "M1")[1:]]
    for cand in candidates:
        if cand in entry:
            return entry[cand]
    return None


def extract_speedup(entry: Dict) -> Optional[float]:
    """Extract speedup factor from various entry schemas."""
    speedup = entry.get("speedup_tile32_vs_tile16",
                        entry.get("speedup", None))
    if speedup is not None:
        return float(speedup)

    t16_ms = entry.get("tile16_mean_ms",
                       entry.get("tile16", {}).get("mean_ms", None))
    t32_ms = entry.get("tile32_mean_ms",
                       entry.get("tile32", {}).get("mean_ms", None))
    if t16_ms and t32_ms and t32_ms > 0:
        return t16_ms / t32_ms
    return None


def run_analysis(
    official_rows: List[Dict],
    synthetic_data: Dict[str, Dict],
    output_dir: Path,
) -> Dict:
    """Run the synthetic vs official comparative analysis."""
    print(f"\n{'='*80}")
    print(f"  SYNTHETIC vs OFFICIAL COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")

    analysis = {
        "synthetic_points": [],
        "official_points": [],
        "comparisons": {},
        "conclusions": {},
    }

    # --- Build synthetic points ---
    for sk, sv in sorted(synthetic_data.items()):
        num_g = sv.get("num_gaussians", 0)
        speedup = sv.get("speedup")
        t16 = sv.get("tile16_mean_ms")
        t32 = sv.get("tile32_mean_ms")
        if speedup is None:
            speedup = extract_speedup(sv)
        if speedup is not None:
            point = {
                "type": "synthetic",
                "label": SYNTHETIC_SCENES.get(sk, {}).get("label", sk),
                "num_gaussians": int(num_g) if num_g else 0,
                "tile16_mean_ms": t16,
                "tile32_mean_ms": t32,
                "speedup": round(speedup, 4),
            }
            analysis["synthetic_points"].append(point)
            print(f"  Synthetic [{sk:10s}]  {num_g:>8,d} GS  "
                  f"speedup={speedup:.4f}x")

    # --- Build official points ---
    for row in official_rows:
        speedup = row.get("speedup_tile32_vs_tile16")
        ng = row.get("num_gaussians", 0)
        t16 = row.get("tile16", {}).get("mean_ms", 0)
        t32 = row.get("tile32", {}).get("mean_ms", 0)
        if speedup is not None:
            point = {
                "type": "official",
                "label": row["scene_id"],
                "dataset": row.get("dataset_family", "Mip-NeRF 360"),
                "num_gaussians": int(ng),
                "tile16_mean_ms": t16,
                "tile32_mean_ms": t32,
                "speedup": round(speedup, 4),
                "resolution": row.get("resolution", ""),
            }
            analysis["official_points"].append(point)
            print(f"  Official [{row['scene_id']:10s}]  {ng:>8,d} GS  "
                  f"speedup={speedup:.4f}x")

    # --- Trend analysis ---
    all_points = analysis["synthetic_points"] + analysis["official_points"]
    if len(all_points) < 2:
        analysis["conclusions"]["trend_analysis"] = "Insufficient data points"
        return analysis

    # Compare trends
    syn_points = analysis["synthetic_points"]
    off_points = analysis["official_points"]

    if len(syn_points) >= 2:
        syn_g = np.array([p["num_gaussians"] for p in syn_points])
        syn_s = np.array([p["speedup"] for p in syn_points])
        if len(syn_g) > 1 and syn_g.ptp() > 0:
            syn_corr = np.corrcoef(syn_g, syn_s)[0, 1]
            analysis["comparisons"]["synthetic_gaussian_speedup_correlation"] = round(float(syn_corr), 4)
            print(f"\n  Synthetic: Gaussian-speedup correlation = {syn_corr:.4f}")
        else:
            syn_corr = None

    if len(off_points) >= 2:
        off_g = np.array([p["num_gaussians"] for p in off_points])
        off_s = np.array([p["speedup"] for p in off_points])
        if len(off_g) > 1 and off_g.ptp() > 0:
            off_corr = np.corrcoef(off_g, off_s)[0, 1]
            analysis["comparisons"]["official_gaussian_speedup_correlation"] = round(float(off_corr), 4)
            print(f"  Official: Gaussian-speedup correlation = {off_corr:.4f}")

    # Speedup range comparison
    if syn_points:
        syn_speedups = [p["speedup"] for p in syn_points]
        analysis["comparisons"]["synthetic_speedup_range"] = {
            "min": round(float(min(syn_speedups)), 4),
            "max": round(float(max(syn_speedups)), 4),
            "mean": round(float(np.mean(syn_speedups)), 4),
            "std": round(float(np.std(syn_speedups)), 4),
        }
    if off_points:
        off_speedups = [p["speedup"] for p in off_points]
        analysis["comparisons"]["official_speedup_range"] = {
            "min": round(float(min(off_speedups)), 4),
            "max": round(float(max(off_speedups)), 4),
            "mean": round(float(np.mean(off_speedups)), 4),
            "std": round(float(np.std(off_speedups)), 4),
        }

    # --- Overlap analysis ---
    if syn_points and off_points:
        syn_min_s = min(p["speedup"] for p in syn_points)
        syn_max_s = max(p["speedup"] for p in syn_points)
        off_min_s = min(p["speedup"] for p in off_points)
        off_max_s = max(p["speedup"] for p in off_points)

        overlap_low = max(syn_min_s, off_min_s)
        overlap_high = min(syn_max_s, off_max_s)
        analysis["comparisons"]["speedup_range_overlap"] = {
            "synthetic_range": [round(syn_min_s, 4), round(syn_max_s, 4)],
            "official_range": [round(off_min_s, 4), round(off_max_s, 4)],
            "overlap_exists": overlap_low <= overlap_high,
            "overlap_interval": [round(overlap_low, 4), round(overlap_high, 4)] if overlap_low <= overlap_high else None,
        }

    # --- Gaussian count ordering check ---
    if off_points:
        off_by_g = sorted(off_points, key=lambda p: p["num_gaussians"])
        monotonic = all(
            off_by_g[i]["speedup"] <= off_by_g[i + 1]["speedup"]
            for i in range(len(off_by_g) - 1)
        )
        analysis["comparisons"]["official_monotonic_gaussian_speedup"] = monotonic
        print(f"\n  Official: Monotonic Gaussian→speedup trend = {monotonic}")

    # --- Conclusions ---
    conclusions = {}

    # Q10: Does synthetic trend replicate on real scenes?
    if syn_points and off_points:
        syn_mean = np.mean([p["speedup"] for p in syn_points])
        off_mean = np.mean([p["speedup"] for p in off_points])
        syn_std = np.std([p["speedup"] for p in syn_points])
        off_std = np.std([p["speedup"] for p in off_points])
        mean_diff = abs(syn_mean - off_mean)
        # If mean speedups within 2 pooled std deviations → consistent
        pooled_std = math.sqrt(syn_std ** 2 + off_std ** 2)
        consistent = mean_diff <= 2 * pooled_std if pooled_std > 0 else None
        if consistent is not None:
            conclusions["Q10_synthetic_trend_on_real"] = (
                "Supported" if consistent else "Inconclusive"
            )
            conclusions["Q10_detail"] = (
                f"Synthetic mean speedup: {syn_mean:.4f}x ± {syn_std:.4f}, "
                f"Official mean speedup: {off_mean:.4f}x ± {off_std:.4f}, "
                f"Pooled diff threshold: {2 * pooled_std:.4f}, "
                f"Actual diff: {mean_diff:.4f}, "
                f"Consistent: {consistent}"
            )
    else:
        conclusions["Q10_synthetic_trend_on_real"] = "Inconclusive"
        conclusions["Q10_detail"] = "Missing synthetic or official data"

    # Q11: Is Gaussian count sufficient to explain real-scene speedup?
    if len(off_points) >= 3:
        corr = off_corr if "off_corr" in dir() or "off_corr" in analysis.get("comparisons", {}) else 0
        # Get actual correlation value
        actual_corr = analysis.get("comparisons", {}).get("official_gaussian_speedup_correlation", None)
        monotonic_flag = analysis.get("comparisons", {}).get("official_monotonic_gaussian_speedup", False)

        if actual_corr is not None and actual_corr > 0.7 and monotonic_flag:
            conclusions["Q11_gaussian_count_explainer"] = "Partially supported"
            conclusions["Q11_detail"] = (
                f"Gaussian-speedup correlation: {actual_corr:.4f}, "
                f"Monotonic: {monotonic_flag}. "
                f"Gaussian count is a strong but not sole predictor."
            )
        elif actual_corr is not None and actual_corr > 0.5:
            conclusions["Q11_gaussian_count_explainer"] = "Inconclusive"
            conclusions["Q11_detail"] = (
                f"Gaussian-speedup correlation: {actual_corr:.4f}. "
                f"Moderate correlation, but limited data points."
            )
        else:
            conclusions["Q11_gaussian_count_explainer"] = "Not supported"
            conclusions["Q11_detail"] = (
                f"Gaussian-speedup correlation: {actual_corr}. "
                f"No clear monotonic relationship."
            )
    else:
        conclusions["Q11_gaussian_count_explainer"] = "Inconclusive"
        conclusions["Q11_detail"] = "Insufficient data points (N < 3)"

    # Q12: Screen-space/density factors
    conclusions["Q12_screen_space_factors"] = "Inconclusive"
    conclusions["Q12_detail"] = (
        "Screen-space projected Gaussian count, tile occupancy, "
        "and depth complexity data not yet collected. "
        "This is a future work item."
    )

    analysis["conclusions"] = conclusions

    # Print conclusions
    print(f"\n{'='*80}")
    print(f"  CONCLUSIONS")
    print(f"{'='*80}")
    for key, value in conclusions.items():
        if key.endswith("_detail"):
            continue
        detail_key = key.replace("Q", "Q") + "_detail"
        detail = conclusions.get(detail_key, "")
        print(f"  {key}: {value}")
        if detail:
            print(f"    {detail}")

    # Save analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "synthetic_vs_official_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\n  Analysis saved: {out_path}")

    return analysis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--official-agg",
        type=Path,
        default=OFFICIAL_AGGREGATED_PATH,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "epic05" / "official" / "statistics",
    )
    args = p.parse_args()

    print(f"Synthetic vs Official Analysis\n")
    print(f"  Loading synthetic results...")

    # Try multiple methods to load synthetic data
    synthetic_data = load_synthetic_results()
    if not synthetic_data:
        synthetic_data = load_synthetic_speed_results_from_optimization()
    if not synthetic_data:
        # Try reading raw optimization result files
        raw_dir = REPO_ROOT / "results" / "epic05" / "raw"
        if raw_dir.exists():
            for fpath in sorted(raw_dir.glob("*.json")):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                # Extract any tile_size comparison data
                for outer_key in data:
                    outer_val = data[outer_key]
                    if isinstance(outer_val, dict):
                        for sk in ("50k", "200k", "400k", "scene_50k",
                                    "scene_200k", "scene_400k"):
                            if sk in outer_val or sk in data:
                                target = outer_val.get(sk, data.get(sk, None))
                                if target and isinstance(target, dict):
                                    tile16 = (target.get("tile16") or
                                              target.get("M1b") or {})
                                    tile32 = (target.get("tile32") or
                                              target.get("M1c") or {})
                                    t16_ms = (tile16.get("mean_ms") or
                                              tile16.get("mean_latency_ms"))
                                    t32_ms = (tile32.get("mean_ms") or
                                              tile32.get("mean_latency_ms"))
                                    if t16_ms and t32_ms:
                                        synthetic_data[sk] = {
                                            "num_gaussians": target.get("num_gaussians", 0) or
                                                             target.get("gaussians", 0),
                                            "tile16_mean_ms": t16_ms,
                                            "tile32_mean_ms": t32_ms,
                                            "speedup": t16_ms / t32_ms if t32_ms > 0 else None,
                                        }

    print(f"  Found {len(synthetic_data)} synthetic entries: "
          f"{list(synthetic_data.keys())}")

    print(f"  Loading official aggregated results from {args.official_agg}...")
    official_rows = load_official_aggregated(args.official_agg)
    print(f"  Found {len(official_rows)} official entries")

    run_analysis(synthetic_data, official_rows, args.output_dir)


if __name__ == "__main__":
    main()
