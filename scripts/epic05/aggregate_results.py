#!/usr/bin/env python3
"""
Aggregate EPIC-05 optimization results.
Consumes raw JSON files from results/epic05/raw/ and produces
summary tables, comparison tables, and machine-readable CSVs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_ablation_results(path: Path) -> Optional[Dict[str, Any]]:
    """Load aggregated ablation results."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def build_ablation_table(agg: Dict[str, Any]) -> str:
    """Build an ablation markdown table."""
    rows = agg["results"]
    baseline = None
    for r in rows:
        if r.get("module_id") == "M0":
            baseline = r
            break

    lines = [
        "## Ablation Results\n",
        f"Scene: {agg.get('scene_key', 'N/A')}  |  Resolution: {agg.get('resolution', 'N/A')}",
        f"Protocol: {agg['protocol']['warmup']} warmup, {agg['protocol']['frames']} frames, {agg['protocol']['repeats']} repeats\n",
        "| Module | Name | Mean (ms) | Δ ms | FPS | Δ FPS % | VRAM (MB) | P99 (ms) | Speedup |",
        "|--------|------|----------:|-----:|----:|--------:|----------:|---------:|--------:|",
    ]

    baseline_mean = baseline["mean_ms"] if baseline else 1.0
    baseline_fps = baseline["mean_fps"] if baseline else 1.0

    for r in rows:
        mod_id = r.get("module_id", "?")
        name = r.get("module_name", "?")
        mean = r["mean_ms"]
        fps = r["mean_fps"]
        vram = r["peak_vram_mb"]
        p99 = r["p99_ms"]
        if mod_id == "M0":
            delta = "-"
            delta_pct = "-"
            speedup = "1.00x"
        else:
            delta = f"{mean - baseline_mean:+.2f}"
            delta_pct = f"{(fps - baseline_fps) / baseline_fps * 100:+.1f}"
            speedup = f"{baseline_mean / mean:.2f}x"
        lines.append(
            f"| {mod_id} | {name} | {mean:.2f} | {delta} | {fps:.1f} | {delta_pct}% | {vram:.0f} | {p99:.2f} | {speedup} |"
        )

    return "\n".join(lines)


def build_interaction_table(agg: Dict[str, Any]) -> str:
    """Build an interaction experiment markdown table."""
    rows = agg["results"]
    baseline = None
    for r in rows:
        if r.get("experiment_id") == "I1" or (not r.get("modules") or "M0" in r.get("modules", [])):
            baseline = r
            break

    lines = [
        "## Interaction Results\n",
        f"Scene: {agg.get('scene_key', 'N/A')}  |  Resolution: {agg.get('resolution', 'N/A')}\n",
        "| Exp ID | Description | Mean (ms) | FPS | VRAM (MB) | P99 (ms) | Speedup |",
        "|--------|-------------|----------:|----:|----------:|---------:|--------:|",
    ]

    baseline_mean = baseline["mean_ms"] if baseline else 1.0

    for r in rows:
        eid = r.get("experiment_id", "?")
        desc = r.get("description", "?")
        mean = r["mean_ms"]
        fps = r["mean_fps"]
        vram = r["peak_vram_mb"]
        p99 = r["p99_ms"]
        speedup = f"{baseline_mean / mean:.2f}x" if baseline_mean else "N/A"
        lines.append(
            f"| {eid} | {desc} | {mean:.2f} | {fps:.1f} | {vram:.0f} | {p99:.2f} | {speedup} |"
        )

    return "\n".join(lines)


def build_fwd_bwd_table(profile: Dict[str, Any]) -> str:
    """Build forward/backward breakdown table."""
    lines = [
        "## Forward / Backward Breakdown\n",
        f"Scene: {profile.get('scene_key', 'N/A')}  |  Resolution: {profile.get('resolution', 'N/A')}\n",
        "| Tile Size | Forward (ms) | Fwd % | Backward (ms) | Bwd % | Total (ms) |",
        "|----------:|------------:|------:|--------------:|------:|-----------:|",
    ]

    for fwd, bwd, total in zip(profile.get("fwd", []), profile.get("bwd", []), profile.get("total", [])):
        ts = fwd["tile_size"]
        fwd_ms = fwd["mean_ms"]
        bwd_ms = bwd["mean_ms"]
        total_ms = total["mean_ms"]
        fwd_pct = fwd_ms / total_ms * 100 if total_ms > 0 else 0
        bwd_pct = bwd_ms / total_ms * 100 if total_ms > 0 else 0
        lines.append(
            f"| {ts} | {fwd_ms:.2f} | {fwd_pct:.1f}% | {bwd_ms:.2f} | {bwd_pct:.1f}% | {total_ms:.2f} |"
        )

    return "\n".join(lines)


def build_overall_table(ablation_rows: List[Dict], interaction_rows: List[Dict]) -> str:
    """Build overall comparison table."""
    lines = [
        "## Overall Speed and Quality\n",
        "| Config | Type | Mean (ms) | FPS | Speedup | VRAM (MB) | P99 (ms) |",
        "|--------|------|----------:|----:|--------:|----------:|---------:|",
    ]

    baseline = None
    for r in ablation_rows:
        if r.get("module_id") == "M0":
            baseline = r
            lines.append(f"| M0 baseline | ablation | {r['mean_ms']:.2f} | {r['mean_fps']:.1f} | 1.00x | {r['peak_vram_mb']:.0f} | {r['p99_ms']:.2f} |")
            break

    if baseline:
        b_mean = baseline["mean_ms"]

        # Best single
        best_single = max(ablation_rows, key=lambda r: r["mean_fps"]) if ablation_rows else None
        if best_single and best_single.get("module_id") != "M0":
            speedup = b_mean / best_single["mean_ms"]
            lines.append(f"| {best_single.get('module_id', '?')} {best_single.get('module_name', '?')} | best-single | {best_single['mean_ms']:.2f} | {best_single['mean_fps']:.1f} | {speedup:.2f}x | {best_single['peak_vram_mb']:.0f} | {best_single['p99_ms']:.2f} |")

        # Best interaction
        best_inter = max(interaction_rows, key=lambda r: r["mean_fps"]) if interaction_rows else None
        if best_inter:
            speedup = b_mean / best_inter["mean_ms"]
            lines.append(f"| {best_inter.get('experiment_id', '?')} {best_inter.get('description', '?')} | best-interaction | {best_inter['mean_ms']:.2f} | {best_inter['mean_fps']:.1f} | {speedup:.2f}x | {best_inter['peak_vram_mb']:.0f} | {best_inter['p99_ms']:.2f} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate EPIC-05 results")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input aggregated results directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown file path")
    parser.add_argument("--scene", type=str, default="50k",
                        help="Scene key")
    parser.add_argument("--resolution", type=str, default="1080p",
                        help="Resolution")
    args = parser.parse_args()

    if args.input_dir:
        agg_dir = Path(args.input_dir)
    else:
        agg_dir = REPO_ROOT / "results" / "epic05" / "aggregated"

    scene = args.scene
    resolution = args.resolution

    # Load ablation
    ablation_path = agg_dir / f"ablation_{scene}_{resolution}.json"
    ablation_agg = load_ablation_results(ablation_path)
    if ablation_agg is None:
        print(f"  No ablation results found at {ablation_path}")
        ablation_table = ""
        ablation_rows = []
    else:
        ablation_table = build_ablation_table(ablation_agg)
        ablation_rows = ablation_agg["results"]

    # Load interaction
    interaction_path = agg_dir / f"interaction_{scene}_{resolution}.json"
    interaction_agg = load_ablation_results(interaction_path)
    if interaction_agg is None:
        print(f"  No interaction results found at {interaction_path}")
        interaction_table = ""
        interaction_rows = []
    else:
        interaction_table = build_interaction_table(interaction_agg)
        interaction_rows = interaction_agg["results"]

    # Load profile
    profile_path = REPO_ROOT / "results" / "epic05" / "profiles" / f"fwd_bwd_profile_{scene}_{resolution}.json"
    profile_data = load_ablation_results(profile_path)
    if profile_data:
        fwd_bwd_table = build_fwd_bwd_table(profile_data)
    else:
        fwd_bwd_table = ""

    # Build overall
    overall_table = build_overall_table(ablation_rows, interaction_rows)

    # Assemble report
    report = [
        f"# EPIC-05 Optimization Results ({resolution}, {scene})",
        f"\n*Generated: {np.datetime_as_string(np.datetime64('now'))}*\n",
        "---\n",
        overall_table,
        "---\n",
        ablation_table,
        "---\n",
        interaction_table,
        "---\n",
        fwd_bwd_table,
        "---\n",
        "## Key Findings\n",
        "See docs/epic05 for full protocol and analysis.\n",
    ]

    report_text = "\n".join(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report_text)
        print(f"  Report written to {out_path}")
    else:
        print(report_text)


if __name__ == "__main__":
    main()
