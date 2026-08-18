#!/usr/bin/env python3
"""
Generate publication-quality charts for the official dataset validation.

Charts:
  1. Speed: bar chart of tile16 vs tile32 latency per scene
  2. Speedup: bar chart of tile32 speedup factors with synthetic overlay
  3. Quality: PSNR delta bar chart per scene
  4. Pareto: Speed vs PSNR scatter (color = scene, shape = tile_size)
  5. Gaussian count vs speedup scatter (synthetic + official markers)
  6. VRAM comparison
  7. Pixel equivalence summary

Usage:
    python scripts/epic05/generate_official_charts.py
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Expected file locations
# ---------------------------------------------------------------------------
OFFICIAL_AGGREGATED = (
    REPO_ROOT / "results" / "epic05" / "official" / "aggregated" / "official_aggregated.json"
)
QUALITY_DIR = REPO_ROOT / "results" / "epic05" / "official" / "quality"
STATISTICS_DIR = REPO_ROOT / "results" / "epic05" / "official" / "statistics"
SYNTHETIC_VS_OFFICIAL = (
    STATISTICS_DIR / "synthetic_vs_official_analysis.json"
)
OUTPUT_DIR = REPO_ROOT / "figures" / "epic05" / "official"


def load_json(path: Path):
    if not path.exists():
        print(f"WARNING: {path} not found")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_speed_chart(rows: List[Dict], output_dir: Path):
    """Generate tile16 vs tile32 latency bar chart."""
    if not rows:
        print("  Skipping speed chart: no data")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping charts")
        return

    scenes = [r["scene_id"] for r in rows]
    t16_ms = [r.get("tile16", {}).get("mean_ms", 0) for r in rows]
    t32_ms = [r.get("tile32", {}).get("mean_ms", 0) for r in rows]

    x = np.arange(len(scenes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, t16_ms, width, label="tile16", color="#4A90D9")
    bars2 = ax.bar(x + width / 2, t32_ms, width, label="tile32", color="#E67E22")

    ax.set_xlabel("Scene")
    ax.set_ylabel("Mean Render Time (ms)")
    ax.set_title("tile16 vs tile32 Latency on Official Mip-NeRF 360 Scenes")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.legend()

    # Annotate speedup
    for i, r in enumerate(rows):
        speedup = r.get("speedup_tile32_vs_tile16")
        if speedup:
            ax.annotate(
                f"{speedup:.3f}x",
                xy=(i + width / 2, t32_ms[i]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", fontsize=9, color="#E67E22", fontweight="bold",
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_dir / "official_speed_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "official_speed_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: official_speed_comparison.png/.pdf")


def generate_speedup_chart(rows: List[Dict], analysis: Optional[Dict],
                           output_dir: Path):
    """Generate speedup bar chart with synthetic overlay."""
    if not rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scenes = [r["scene_id"] for r in rows]
    speedups = [r.get("speedup_tile32_vs_tile16", 0) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenes))
    bars = ax.bar(x, speedups, color="#E67E22", alpha=0.8)

    # Add synthetic overlay
    if analysis:
        syn_points = analysis.get("synthetic_points", [])
        if syn_points:
            syn_labels = [p["label"] for p in syn_points]
            syn_speeds = [p["speedup"] for p in syn_points]
            syn_x = np.arange(len(scenes), len(scenes) + len(syn_points))
            syn_bars = ax.bar(
                syn_x, syn_speeds, color="#3498DB", alpha=0.8,
                label="Synthetic (controlled)"
            )
            all_labels = scenes + syn_labels
            all_x = np.arange(len(all_labels))
            ax.set_xticks(all_x)
            ax.set_xticklabels(all_labels, rotation=30, ha="right")

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No speedup (1.0x)")
    ax.set_ylabel("Speedup Factor (tile16 / tile32)")
    ax.set_title("tile32 Speedup: Official Scenes vs Synthetic Workloads")
    ax.legend()

    # Annotate
    for i, (scene, sp) in enumerate(zip(scenes, speedups)):
        ax.annotate(f"{sp:.3f}x", xy=(x[i], sp),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_dir / "official_speedup_with_synthetic.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "official_speedup_with_synthetic.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: official_speedup_with_synthetic.png/.pdf")


def generate_gaussian_vs_speedup_chart(
    rows: List[Dict], analysis: Optional[Dict], output_dir: Path
):
    """Scatter: Gaussian count vs speedup, synthetic + official markers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Official points
    g_off = [r["num_gaussians"] for r in rows]
    s_off = [r.get("speedup_tile32_vs_tile16", 0) for r in rows]
    labels_off = [r["scene_id"] for r in rows]

    ax.scatter(g_off, s_off, marker="o", s=120, color="#E67E22",
               zorder=5, label="Official Mip-NeRF 360")
    for i, label in enumerate(labels_off):
        ax.annotate(label, (g_off[i], s_off[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Synthetic points
    if analysis:
        syn_points = analysis.get("synthetic_points", [])
        if syn_points:
            g_syn = [p["num_gaussians"] for p in syn_points]
            s_syn = [p["speedup"] for p in syn_points]
            syn_labels = [p["label"] for p in syn_points]
            ax.scatter(g_syn, s_syn, marker="^", s=120, color="#3498DB",
                       zorder=5, label="Synthetic (controlled)")
            for i, label in enumerate(syn_labels):
                ax.annotate(label, (g_syn[i], s_syn[i]),
                            xytext=(5, -15), textcoords="offset points",
                            fontsize=9, color="#3498DB")

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Gaussian Count")
    ax.set_ylabel("Speedup (tile16 / tile32)")
    ax.set_title("Gaussian Count vs tile32 Speedup: Synthetic and Official Scenes")
    ax.legend()
    ax.set_xscale("log")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_dir / "gaussian_count_vs_speedup.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "gaussian_count_vs_speedup.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: gaussian_count_vs_speedup.png/.pdf")


def generate_quality_chart(quality_results: List[Dict], output_dir: Path):
    """PSNR/SSIM/LPIPS delta bar charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Collect quality data
    scenes = []
    psnr16, psnr32 = [], []
    ssim16, ssim32 = [], []
    lpips16, lpips32 = [], []

    for qr in quality_results:
        q16 = qr.get("tile_quality", {}).get("tile16", {})
        q32 = qr.get("tile_quality", {}).get("tile32", {})
        if "mean_psnr_db" not in q16 or "mean_psnr_db" not in q32:
            continue
        scenes.append(qr["scene_id"])
        psnr16.append(q16["mean_psnr_db"])
        psnr32.append(q32["mean_psnr_db"])
        ssim16.append(q16["mean_ssim"])
        ssim32.append(q32["mean_ssim"])
        lpips16.append(q16["mean_lpips"])
        lpips32.append(q32["mean_lpips"])

    if not scenes:
        print("  Skipping quality chart: no quality data with GT")
        return

    x = np.arange(len(scenes))
    width = 0.35

    # PSNR chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, psnr16, width, label="tile16", color="#4A90D9")
    ax.bar(x + width / 2, psnr32, width, label="tile32", color="#E67E22")
    ax.set_xlabel("Scene")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR Comparison: tile16 vs tile32 vs Ground Truth")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.legend()
    for i, (s16, s32) in enumerate(zip(psnr16, psnr32)):
        delta = s32 - s16
        ax.annotate(
            f"Δ={delta:+.3f}dB",
            xy=(i + width / 2, psnr32[i]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=8,
            color="green" if delta >= -0.1 else "red",
            fontweight="bold",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_dir / "official_psnr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: official_psnr_comparison.png")

    # SSIM chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ssim16, width, label="tile16", color="#4A90D9")
    ax.bar(x + width / 2, ssim32, width, label="tile32", color="#E67E22")
    ax.set_xlabel("Scene")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM Comparison: tile16 vs tile32 vs Ground Truth")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.legend()
    for i, (s16, s32) in enumerate(zip(ssim16, ssim32)):
        delta = s32 - s16
        ax.annotate(
            f"Δ={delta:+.6f}",
            xy=(i + width / 2, ssim32[i]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=8,
        )
    plt.tight_layout()
    fig.savefig(output_dir / "official_ssim_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: official_ssim_comparison.png")

    # LPIPS chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, lpips16, width, label="tile16", color="#4A90D9")
    ax.bar(x + width / 2, lpips32, width, label="tile32", color="#E67E22")
    ax.set_xlabel("Scene")
    ax.set_ylabel("LPIPS")
    ax.set_title("LPIPS Comparison: tile16 vs tile32 vs Ground Truth")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.legend()
    for i, (s16, s32) in enumerate(zip(lpips16, lpips32)):
        delta = s32 - s16
        ax.annotate(
            f"Δ={delta:+.6f}",
            xy=(i + width / 2, lpips32[i]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=8,
        )
    plt.tight_layout()
    fig.savefig(output_dir / "official_lpips_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: official_lpips_comparison.png")


def generate_pareto_chart(
    speed_rows: List[Dict], quality_results: List[Dict], output_dir: Path
):
    """Speed (FPS) vs PSNR scatter plot: tile16 and tile32."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Merge speed and quality data
    for row in speed_rows:
        scene_id = row["scene_id"]
        for qr in quality_results:
            if qr["scene_id"] != scene_id:
                continue
            q16 = qr.get("tile_quality", {}).get("tile16", {})
            q32 = qr.get("tile_quality", {}).get("tile32", {})
            t16_fps = row.get("tile16", {}).get("mean_fps", 0)
            t32_fps = row.get("tile32", {}).get("mean_fps", 0)

            if "mean_psnr_db" in q16 and t16_fps:
                ax.scatter(t16_fps, q16["mean_psnr_db"],
                           marker="o", s=120, label=f"{scene_id} tile16" if row == speed_rows[0] else "")
                ax.annotate(
                    f"{scene_id}_t16",
                    (t16_fps, q16["mean_psnr_db"]),
                    fontsize=7, alpha=0.7,
                )
            if "mean_psnr_db" in q32 and t32_fps:
                ax.scatter(t32_fps, q32["mean_psnr_db"],
                           marker="^", s=120, label=f"{scene_id} tile32" if row == speed_rows[0] else "")
                ax.annotate(
                    f"{scene_id}_t32",
                    (t32_fps, q32["mean_psnr_db"]),
                    fontsize=7, alpha=0.7,
                )

    ax.set_xlabel("Mean FPS (higher is better)")
    ax.set_ylabel("PSNR vs GT (higher is better)")
    ax.set_title("Speed-PSNR Pareto: tile16 vs tile32 on Official Scenes")
    ax.legend(loc="lower right")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_dir / "official_speed_vs_psnr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: official_speed_vs_psnr.png")


def generate_vram_chart(rows: List[Dict], output_dir: Path):
    """VRAM comparison: tile16 vs tile32."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scenes = [r["scene_id"] for r in rows]
    v16 = [r.get("tile16", {}).get("peak_vram_mb", 0) for r in rows]
    v32 = [r.get("tile32", {}).get("peak_vram_mb", 0) for r in rows]

    x = np.arange(len(scenes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, v16, width, label="tile16", color="#4A90D9")
    ax.bar(x + width / 2, v32, width, label="tile32", color="#E67E22")
    ax.set_xlabel("Scene")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Peak VRAM Comparison: tile16 vs tile32")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.legend()

    for i, (v1, v2) in enumerate(zip(v16, v32)):
        delta = v2 - v1
        ax.annotate(
            f"Δ={delta:+.0f}MB",
            xy=(i + width / 2, v2),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=9,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_dir / "official_vram_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: official_vram_comparison.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")

    agg = load_json(OFFICIAL_AGGREGATED)
    rows = agg.get("rows", []) if agg else []
    print(f"  Aggregated results: {len(rows)} scenes")

    analysis = load_json(SYNTHETIC_VS_OFFICIAL)
    print(f"  Synthetic vs official analysis: {'available' if analysis else 'missing'}")

    # Load quality results
    quality_results = []
    if QUALITY_DIR.exists():
        for fpath in sorted(QUALITY_DIR.glob("quality_*.json")):
            qr = load_json(fpath)
            if qr:
                quality_results.append(qr)
    print(f"  Quality results: {len(quality_results)} scenes")

    # Generate charts
    print("\nGenerating charts...")
    generate_speed_chart(rows, output_dir)
    generate_speedup_chart(rows, analysis, output_dir)
    generate_gaussian_vs_speedup_chart(rows, analysis, output_dir)
    generate_quality_chart(quality_results, output_dir)
    generate_pareto_chart(rows, quality_results, output_dir)
    generate_vram_chart(rows, output_dir)

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
