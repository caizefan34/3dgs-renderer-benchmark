#!/usr/bin/env python3
"""
Generate charts from EPIC-05 optimization results.
Produces publication-quality figures for the research report.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Lazy-load matplotlib to avoid headless issues
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def chart_ablation_waterfall(ablation_agg: Dict, output_path: Path):
    """Ablation waterfall chart showing incremental speedup."""
    rows = ablation_agg["results"]
    baseline = next((r for r in rows if r.get("module_id") == "M0"), None)
    if not baseline:
        return

    b_mean = baseline["mean_ms"]

    # Sort by speedup descending
    non_baseline = [r for r in rows if r.get("module_id") != "M0"]
    non_baseline.sort(key=lambda r: b_mean / r["mean_ms"], reverse=True)

    labels = [f"{r['module_id']} {r['module_name']}" for r in non_baseline]
    speedups = [b_mean / r["mean_ms"] for r in non_baseline]
    colors = ["green" if s > 1.0 else "red" if s < 1.0 else "gray" for s in speedups]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), speedups, color=colors)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5, label="Baseline (1.0x)")
    ax.set_xlabel("Speedup vs Baseline (higher is better)")
    ax.set_title(f"Ablation Waterfall ({ablation_agg.get('scene_key', '?')}, {ablation_agg.get('resolution', '?')})")

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}x", va="center", fontsize=8)

    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved waterfall chart: {output_path}")


def chart_fwd_bwd_breakdown(profile: Dict, output_path: Path):
    """Forward/backward breakdown bar chart."""
    fwd_data = profile.get("fwd", [])
    if not fwd_data:
        return

    tile_sizes = [d["tile_size"] for d in fwd_data]
    fwd_ms = [d["mean_ms"] for d in fwd_data]
    bwd_ms = [d["mean_ms"] for d in profile.get("bwd", [])]
    # Use same tile_size alignment
    bwd_ms_aligned = [d["mean_ms"] for d in profile.get("bwd", [])]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tile_sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, fwd_ms, width, label="Forward", color="steelblue")
    bars2 = ax.bar(x + width/2, bwd_ms_aligned, width, label="Backward", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels([f"tile {ts}" for ts in tile_sizes])
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Forward vs Backward ({profile.get('scene_key', '?')}, {profile.get('resolution', '?')})")
    ax.legend()

    # Add % labels
    for b1, b2, fw, bw in zip(bars1, bars2, fwd_ms, bwd_ms_aligned):
        total = fw + bw
        if total > 0:
            ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.1,
                    f"{fw/total*100:.0f}%", ha="center", fontsize=9)
            ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.1,
                    f"{bw/total*100:.0f}%", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fwd/bwd chart: {output_path}")


def chart_interaction_heatmap(interaction_agg: Dict, output_path: Path):
    """Interaction effect heatmap."""
    rows = interaction_agg["results"]
    baseline = next((r for r in rows if r.get("experiment_id") == "I1" or not r.get("modules") or "M0" in r.get("modules", [])), None)
    if not baseline or not rows:
        return

    b_mean = baseline["mean_ms"]

    # Build matrix
    exp_ids = [r.get("experiment_id", "?") for r in rows]
    speedups = np.array([b_mean / r["mean_ms"] for r in rows])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(speedups.reshape(1, -1), cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.5)

    ax.set_xticks(range(len(exp_ids)))
    ax.set_xticklabels(exp_ids, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([0])
    ax.set_yticklabels(["Speedup"], fontsize=10)

    for i, val in enumerate(speedups):
        color = "white" if abs(val - 1.0) > 0.2 else "black"
        ax.text(i, 0, f"{val:.2f}x", ha="center", va="center", fontsize=9, color=color)

    ax.set_title(f"Interaction Heatmap ({interaction_agg.get('scene_key', '?')}, {interaction_agg.get('resolution', '?')})")
    plt.colorbar(im, ax=ax, label="Speedup vs Baseline")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved interaction heatmap: {output_path}")


def chart_kernel_time_distribution(profile: Dict, output_path: Path):
    """Kernel time distribution pie/bar."""
    # Since we don't have per-kernel profile data from Nsight,
    # we show the fwd/bwd split
    fwd_data = profile.get("fwd", [])
    bwd_data = profile.get("bwd", [])
    if not fwd_data or not bwd_data:
        return

    # Use tile_size=16 (default)
    fwd16 = next((d for d in fwd_data if d["tile_size"] == 16), fwd_data[0])
    bwd16 = next((d for d in bwd_data if d["tile_size"] == 16), bwd_data[0])

    labels = ["Forward Rasterization", "Backward Rasterization"]
    sizes = [fwd16["mean_ms"], bwd16["mean_ms"]]
    colors = ["steelblue", "coral"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%",
                                       startangle=90, colors=colors)
    ax.set_title(f"Kernel Time Distribution (tile 16)\n{profile.get('scene_key', '?')}, {profile.get('resolution', '?')}")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved kernel distribution: {output_path}")


def chart_vram_comparison(ablation_agg: Dict, output_path: Path):
    """Peak VRAM comparison bar chart."""
    rows = ablation_agg["results"]
    if not rows:
        return

    baseline = next((r for r in rows if r.get("module_id") == "M0"), None)
    b_vram = baseline["peak_vram_mb"] if baseline else 0

    non_baseline = [r for r in rows if r.get("module_id") != "M0"]
    non_baseline.sort(key=lambda r: r["peak_vram_mb"], reverse=True)

    labels = [f"{r['module_id']} {r['module_name']}" for r in non_baseline]
    vrams = [r["peak_vram_mb"] for r in non_baseline]
    colors = ["green" if v <= b_vram * 1.05 else "red" if v > b_vram * 1.2 else "orange" for v in vrams]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), vrams, color=colors)
    ax.axhline(y=b_vram, color="black", linestyle="--", alpha=0.5, label=f"Baseline ({b_vram:.0f} MB)")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Peak VRAM Comparison")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved VRAM chart: {output_path}")


def main():
    base_dir = REPO_ROOT / "results" / "epic05"
    aggregated_dir = base_dir / "aggregated"
    profiles_dir = base_dir / "profiles"
    figures_dir = REPO_ROOT / "figures" / "epic05"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for scene in ["50k", "200k", "400k"]:
        for res in ["720p", "1080p", "4k"]:
            label = f"{scene}_{res}"

            # Ablation
            ablation = load_json(aggregated_dir / f"ablation_{label}.json")
            if ablation:
                chart_ablation_waterfall(ablation, figures_dir / f"ablation_waterfall_{label}.png")
                chart_vram_comparison(ablation, figures_dir / f"vram_comparison_{label}.png")

            # Interaction
            interaction = load_json(aggregated_dir / f"interaction_{label}.json")
            if interaction:
                chart_interaction_heatmap(interaction, figures_dir / f"interaction_heatmap_{label}.png")

            # Profile
            profile = load_json(profiles_dir / f"fwd_bwd_profile_{label}.json")
            if profile:
                chart_fwd_bwd_breakdown(profile, figures_dir / f"fwd_bwd_breakdown_{label}.png")
                chart_kernel_time_distribution(profile, figures_dir / f"kernel_distribution_{label}.png")

    print(f"\nAll charts saved to {figures_dir}/")


if __name__ == "__main__":
    main()
