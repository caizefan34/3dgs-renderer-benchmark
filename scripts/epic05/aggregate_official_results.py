#!/usr/bin/env python3
"""
Aggregate raw official validation results into a unified machine-readable table.

Scans results/epic05/official/raw/ for JSON result files, computes per-scene
speedups, and writes aggregated CSV + JSON tables.

Usage:
    python scripts/epic05/aggregate_official_results.py
    python scripts/epic05/aggregate_official_results.py --input-dir results/epic05/official/raw
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_raw_results(input_dir: Path) -> list:
    """Load all official validation raw JSON files from the input directory."""
    results = []
    if not input_dir.exists():
        print(f"WARNING: Input directory not found: {input_dir}")
        return results
    for fpath in sorted(input_dir.glob("*.json")):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Skipping {fpath.name}: {e}")
    return results


def aggregate(results: list) -> list:
    """Compute aggregated scene-level table with speedup and quality fields."""
    rows = []
    for run in results:
        protocol = run.get("protocol", {})
        scenes = run.get("scenes", {})
        for scene_id, scene_data in scenes.items():
            if "error" in scene_data:
                continue
            scene_info = scene_data.get("scene_info", {})
            tile_results = scene_data.get("tile_results", {})

            num_gaussians = scene_info.get("num_gaussians", 0)
            resolution = scene_info.get("resolution_used", [1920, 1080])
            resolution_label = f"{resolution[0]}x{resolution[1]}"

            # Speed data per tile size
            tile_data = {}
            for tile_key, tr in tile_results.items():
                ts = tr.get("tile_size", 0)
                tile_data[ts] = {
                    "mean_ms": tr["mean_ms"],
                    "median_ms": tr["median_ms"],
                    "std_ms": tr["std_ms"],
                    "p1_ms": tr.get("p1_ms", 0),
                    "p5_ms": tr.get("p5_ms", 0),
                    "p95_ms": tr.get("p95_ms", 0),
                    "p99_ms": tr.get("p99_ms", 0),
                    "mean_fps": tr["mean_fps"],
                    "median_fps": tr.get("median_fps", 0),
                    "peak_vram_mb": tr["peak_vram_mb"],
                    "avg_vram_mb": tr["avg_vram_mb"],
                }

            row = {
                "scene_id": scene_id,
                "dataset_family": scene_info.get("dataset_family", "Mip-NeRF 360"),
                "num_gaussians": num_gaussians,
                "resolution": resolution_label,
                "camera_count": scene_info.get("camera_count", 0),
                "protocol": {
                    "warmup_frames": protocol.get("warmup_frames", 0),
                    "measured_frames_per_repeat": protocol.get("measured_frames_per_repeat", 0),
                    "repeats": protocol.get("repeats", 0),
                },
            }

            # tile16 performance (baseline)
            if 16 in tile_data:
                row["tile16"] = tile_data[16]
            if 32 in tile_data:
                row["tile32"] = tile_data[32]
            if 8 in tile_data:
                row["tile8"] = tile_data[8]

            # Speedup calculations
            if 16 in tile_data and 32 in tile_data:
                t16 = tile_data[16]["mean_ms"]
                t32 = tile_data[32]["mean_ms"]
                row["speedup_tile32_vs_tile16"] = round(t16 / t32, 4) if t32 > 0 else None
                f16 = tile_data[16]["mean_fps"]
                f32 = tile_data[32]["mean_fps"]
                row["fps_speedup_tile32_vs_tile16"] = round(
                    ((f32 - f16) / f16) * 100, 2
                ) if f16 > 0 else None
                row["vram_delta_mb"] = round(
                    tile_data[32]["peak_vram_mb"] - tile_data[16]["peak_vram_mb"], 1
                )

            if 8 in tile_data and 16 in tile_data:
                t8 = tile_data[8]["mean_ms"]
                t16 = tile_data[16]["mean_ms"]
                row["speedup_tile16_vs_tile8"] = round(t8 / t16, 4) if t16 > 0 else None

            rows.append(row)

    return rows


def export_aggregated_csv(rows: list, output_path: Path) -> None:
    """Export aggregated results as CSV."""
    import csv

    if not rows:
        print("  No data to export")
        return

    fieldnames = [
        "scene_id", "dataset_family", "num_gaussians", "resolution",
        "camera_count",
        # tile16
        "tile16_mean_ms", "tile16_median_ms", "tile16_std_ms",
        "tile16_p99_ms", "tile16_mean_fps", "tile16_peak_vram_mb",
        # tile32
        "tile32_mean_ms", "tile32_median_ms", "tile32_std_ms",
        "tile32_p99_ms", "tile32_mean_fps", "tile32_peak_vram_mb",
        # tile8
        "tile8_mean_ms", "tile8_median_ms", "tile8_p99_ms",
        "tile8_mean_fps", "tile8_peak_vram_mb",
        # comparisons
        "speedup_tile32_vs_tile16",
        "fps_speedup_tile32_vs_tile16_pct",
        "vram_delta_mb",
        "speedup_tile16_vs_tile8",
    ]

    def _flatten(row):
        flat = {
            "scene_id": row["scene_id"],
            "dataset_family": row["dataset_family"],
            "num_gaussians": row["num_gaussians"],
            "resolution": row["resolution"],
            "camera_count": row.get("camera_count", 0),
        }
        for ts in (8, 16, 32):
            key = f"tile{ts}"
            if key in row:
                d = row[key]
                flat[f"tile{ts}_mean_ms"] = d["mean_ms"]
                flat[f"tile{ts}_median_ms"] = d["median_ms"]
                flat[f"tile{ts}_std_ms"] = d.get("std_ms", "")
                flat[f"tile{ts}_p99_ms"] = d.get("p99_ms", "")
                flat[f"tile{ts}_mean_fps"] = d["mean_fps"]
                flat[f"tile{ts}_peak_vram_mb"] = d["peak_vram_mb"]
            else:
                for suffix in ("mean_ms", "median_ms", "std_ms", "p99_ms",
                               "mean_fps", "peak_vram_mb"):
                    flat[f"tile{ts}_{suffix}"] = ""
        flat["speedup_tile32_vs_tile16"] = row.get("speedup_tile32_vs_tile16", "")
        flat["fps_speedup_tile32_vs_tile16_pct"] = row.get(
            "fps_speedup_tile32_vs_tile16", ""
        )
        flat["vram_delta_mb"] = row.get("vram_delta_mb", "")
        flat["speedup_tile16_vs_tile8"] = row.get("speedup_tile16_vs_tile8", "")
        return flat

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten(row))
    print(f"  Exported: {output_path}")


def export_aggregated_json(rows: list, output_path: Path) -> None:
    """Export aggregated results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "benchmark_type": "official_real_scene_validation",
                "aggregated_from": "results/epic05/official/raw/",
                "rows": rows,
            },
            f, indent=2, ensure_ascii=False, allow_nan=False,
        )
    print(f"  Exported: {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "results" / "epic05" / "official" / "raw",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "epic05" / "official" / "aggregated",
    )
    args = p.parse_args()

    print(f"Aggregating official validation results...")
    print(f"  Input:  {args.input_dir}")

    raw_results = load_raw_results(args.input_dir)
    print(f"  Found {len(raw_results)} raw result file(s)")

    if not raw_results:
        print("  No results to aggregate.")
        return

    rows = aggregate(raw_results)
    print(f"  Aggregated {len(rows)} scene entries")

    export_aggregated_json(rows, args.output_dir / "official_aggregated.json")
    export_aggregated_csv(rows, args.output_dir / "official_aggregated.csv")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  OFFICIAL VALIDATION AGGREGATED RESULTS")
    print(f"{'='*90}")
    print(f"  {'Scene':12s} {'Gaussians':>10s} {'Res':10s} "
          f"{'tile16(ms)':>10s} {'tile32(ms)':>10s} {'Speedup':>8s} "
          f"{'VRAM16':>8s} {'VRAM32':>8s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} "
          f"{'-'*8} {'-'*8}")
    for row in rows:
        t16 = row.get("tile16", {}).get("mean_ms", 0)
        t32 = row.get("tile32", {}).get("mean_ms", 0)
        v16 = row.get("tile16", {}).get("peak_vram_mb", 0)
        v32 = row.get("tile32", {}).get("peak_vram_mb", 0)
        speedup = row.get("speedup_tile32_vs_tile16", "")
        speedup_str = f"{speedup:.4f}x" if speedup else "N/A"
        print(f"  {row['scene_id']:12s} {row['num_gaussians']:>10,d} "
              f"{row['resolution']:>10s} {t16:>10.2f} {t32:>10.2f} "
              f"{speedup_str:>8s} {v16:>8.0f} {v32:>8.0f}")


if __name__ == "__main__":
    main()
