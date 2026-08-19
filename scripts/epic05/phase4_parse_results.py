#!/usr/bin/env python3
"""Parse Phase 4 results correctly from multiple files."""
import json
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\36570\3dgs-renderer-benchmark")
phase4_dir = REPO_ROOT / "results" / "epic05" / "phase4"

results = {}  # scene -> resolution -> tile_data

for fpath in sorted(phase4_dir.glob("bicycle_rerun_*.json")):
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    res_label = data.get("resolution_label", "?")
    for sid, sv in data.get("scenes", {}).items():
        key = f"{sid}_{res_label}"
        if key not in results:
            results[key] = {
                "scene": sid,
                "resolution": res_label,
                "gaussians": sv.get("scene_info", {}).get("num_gaussians", 0),
            }
        for tk, tv in sv.get("tile_results", {}).items():
            results[key][tk] = {
                "stable_mean_ms": tv.get("stable_mean_ms"),
                "median_ms": tv.get("median_ms"),
                "p99_ms": tv.get("p99_ms"),
                "outliers": tv.get("outlier_analysis", {}).get("count_severe_outliers", 0),
            }

# Print summary
print("Phase 4 Official Scene Results (RTX 5070 Laptop)")
print("=" * 80)

for key in sorted(results.keys()):
    s = results[key]
    print(f"\nScene: {s['scene']} ({s['resolution']}, {s['gaussians']:,} gaussians)")
    print(f"{'Tile':<8} {'Stable Mean':<15} {'Median':<12} {'P99':<12} {'Outliers':<10}")
    print("-" * 60)
    for tk in ["tile8", "tile16", "tile32"]:
        t = s.get(tk, {})
        if t and t.get("stable_mean_ms"):
            print(f"{tk:<8} {t['stable_mean_ms']:<15.2f} {t['median_ms']:<12.2f} {t['p99_ms']:<12.2f} {t['outliers']:<10d}")

print("\n\nSpeedup Summary:")
for key in sorted(results.keys()):
    s = results[key]
    t16 = s.get("tile16", {}).get("stable_mean_ms")
    t32 = s.get("tile32", {}).get("stable_mean_ms")
    t8 = s.get("tile8", {}).get("stable_mean_ms")
    if t16 and t32:
        sp = t16 / t32
        winner = "tile16" if sp > 1.0 else "tile32"
        print(f"  {key}: tile16={t16:.2f}ms tile32={t32:.2f}ms ratio={sp:.4f}x [{winner} wins]")
    if t16 and t8:
        sp8 = t16 / t8
        print(f"  {key}: tile8={t8:.2f}ms tile16={t16:.2f}ms ratio={sp8:.4f}x")
