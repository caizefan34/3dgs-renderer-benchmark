#!/usr/bin/env python3
"""Generate the 4 required figures from results.json."""
import json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent / "reports" / "accutile30k"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIPNERF360 = ["bicycle", "bonsai", "counter", "flowers", "garden",
              "kitchen", "room", "stump", "treehill"]
TANKSTEMPLES = ["train", "truck"]
DEEPBLENDING = ["drjohnson", "playroom"]
ALL = MIPNERF360 + TANKSTEMPLES + DEEPBLENDING


def load():
    with open(REPORT_DIR / "results.json") as f:
        return json.load(f)


def fig_quality_delta(data):
    """Bar chart of ΔPSNR per scene."""
    per = data["per_scene"]
    scenes = [e["scene"] for e in per if e.get("dpsnr") is not None]
    deltas = [e["dpsnr"] for e in per if e.get("dpsnr") is not None]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(scenes)), deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔPSNR (dB) = B1A − B1")
    ax.set_title("Quality Delta: B1A vs B1 (30K, per scene)")
    ax.grid(axis="y", alpha=0.3)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{d:+.2f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "quality-delta.png", dpi=150)
    plt.close()
    print("  saved quality-delta.png")


def fig_training_speedup(data):
    """Bar chart of full-training speedup per scene + geomean line."""
    per = data["per_scene"]
    scenes = [e["scene"] for e in per if e.get("training_speedup") is not None]
    speedups = [e["training_speedup"] for e in per if e.get("training_speedup") is not None]
    colors = ["#2ca02c" if s > 1.0 else "#d62728" for s in speedups]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(scenes)), speedups, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, rotation=45, ha="right")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="no change (1.0×)")
    gm = data["dataset_aggregation"]["all_13"].get("geomean_training_speedup")
    if gm:
        ax.axhline(gm, color="#1f77b4", linewidth=1.5, linestyle=":", label=f"geomean {gm:.3f}×")
    ax.set_ylabel("Training speedup = T_B1 / T_B1A")
    ax.set_title("Full-Training Speedup: B1A vs B1 (30K, per scene)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s:.2f}×", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training-speedup.png", dpi=150)
    plt.close()
    print("  saved training-speedup.png")


def fig_convergence(data):
    """PSNR vs iteration for B1 and B1A across all scenes (2-panel: B1 left, B1A right, or overlaid)."""
    per = data["per_scene"]
    iters = [5000, 10000, 15000, 20000, 25000, 30000]
    fig, ax = plt.subplots(figsize=(12, 6))
    for e in per:
        traj = e.get("psnr_trajectory", {})
        b1_vals = [traj.get(str(it), {}).get("b1") for it in iters]
        b1a_vals = [traj.get(str(it), {}).get("b1a") for it in iters]
        if any(v is not None for v in b1_vals):
            ax.plot(iters, b1_vals, "o-", alpha=0.5, linewidth=1, markersize=3, label=f"{e['scene']} B1")
        if any(v is not None for v in b1a_vals):
            ax.plot(iters, b1a_vals, "s--", alpha=0.5, linewidth=1, markersize=3, label=f"{e['scene']} B1A")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Convergence: PSNR vs Iteration (B1 solid, B1A dashed)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=6, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "convergence.png", dpi=150)
    plt.close()
    print("  saved convergence.png")


def fig_gaussian_count(data):
    """N_GS(t) for B1 vs B1A across all scenes."""
    per = data["per_scene"]
    iters = [5000, 10000, 15000, 20000, 25000, 30000]
    fig, ax = plt.subplots(figsize=(12, 6))
    for e in per:
        traj = e.get("ngs_trajectory", {})
        b1_vals = [traj.get(str(it), {}).get("b1") for it in iters]
        b1a_vals = [traj.get(str(it), {}).get("b1a") for it in iters]
        if any(v is not None for v in b1_vals):
            ax.plot(iters, [v/1000 if v else None for v in b1_vals], "o-", alpha=0.5, linewidth=1, markersize=3, label=f"{e['scene']} B1")
        if any(v is not None for v in b1a_vals):
            ax.plot(iters, [v/1000 if v else None for v in b1a_vals], "s--", alpha=0.5, linewidth=1, markersize=3, label=f"{e['scene']} B1A")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gaussian count (thousands)")
    ax.set_title("Topology: N_GS(t) (B1 solid, B1A dashed)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=6, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gaussian-count.png", dpi=150)
    plt.close()
    print("  saved gaussian-count.png")


def main():
    if not (REPORT_DIR / "results.json").exists():
        print("results.json not found; run b1a_13scene_analyze.py first")
        sys.exit(1)
    data = load()
    fig_quality_delta(data)
    fig_training_speedup(data)
    fig_convergence(data)
    fig_gaussian_count(data)
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
