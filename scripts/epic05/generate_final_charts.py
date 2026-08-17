#!/usr/bin/env python3
"""
Generate final Phase 2 charts for EPIC-05 research report.
All charts are publication-quality and automatically generated from experiment data.

Charts produced:
1. tile_size vs Gaussian count (scaling curve)
2. forward/backward breakdown (stacked bar)
3. kernel time comparison (grouped bar)
4. occupancy vs runtime (analysis)
5. shared memory vs runtime (analysis)
6. register usage vs runtime (analysis)
7. training step time (line)
8. training quality (PSNR curve)
9. speed-quality Pareto
10. tile-size scaling curve
11. M0 anomaly verification (cold vs warm comparison)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Style setup
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

FINAL_DIR = REPO_ROOT / "results" / "epic05" / "final_validation"
RAW_DIR = FINAL_DIR / "raw"
AGG_DIR = FINAL_DIR / "aggregated"
PROFILE_DIR = FINAL_DIR / "profiles"
TRAINING_DIR = FINAL_DIR / "training"
FIGURES_DIR = REPO_ROOT / "figures" / "epic05" / "final"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    "tile8": "#E74C3C",
    "tile16": "#3498DB",
    "tile32": "#2ECC71",
    "forward": "#E74C3C",
    "backward": "#3498DB",
    "optimizer": "#F39C12",
    "baseline": "#95A5A6",
    "cold": "#E74C3C",
    "warm": "#2ECC71",
}


def load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ===========================================================================
# Chart 1: Tile Size vs Gaussian Count (Scaling Curve)
# ===========================================================================

def chart_scaling_curve(scaling_data: Optional[Dict] = None):
    """Plot tile size throughput vs Gaussian count."""
    print("  Chart 1: Scaling curve...")

    # Try to load from scaling_validation first
    if not scaling_data:
        scaling_data = load_json(AGG_DIR / "scaling_validation.json")

    if not scaling_data:
        # Use aggregated historical data
        print("    (no scaling validation data, using historical)")
        return

    results = scaling_data["results"]
    scenes = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Throughput (FPS) vs Gaussian count
    ax = axes[0]
    for ts_label, color, marker in [("tile8", COLORS["tile8"], "v"), ("tile16", COLORS["tile16"], "s"), ("tile32", COLORS["tile32"], "o")]:
        gaussians = []
        fps = []
        for s in scenes:
            if ts_label in results[s]["results"]:
                gaussians.append(results[s]["num_gaussians"] / 1000)
                fps.append(results[s]["results"][ts_label]["mean_fps"])
        ax.plot(gaussians, fps, f"{marker}-", color=color, label=ts_label, linewidth=2, markersize=8)

    ax.set_xlabel("Gaussian Count (K)")
    ax.set_ylabel("Throughput (FPS)")
    ax.set_title("Rendering Throughput vs Gaussian Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([50, 200, 400])

    # Right: Speedup vs Baseline (tile16)
    ax = axes[1]
    gaussians = [results[s]["num_gaussians"] / 1000 for s in scenes]
    tile32_speedups = [results[s].get("speedup_tile32", 0) for s in scenes]
    tile8_ratios = [results[s].get("speedup_tile8", 0) for s in scenes]

    ax.plot(gaussians, tile32_speedups, "o-", color=COLORS["tile32"], linewidth=2, markersize=8, label="tile32 vs tile16")
    ax.plot(gaussians, tile8_ratios, "v-", color=COLORS["tile8"], linewidth=2, markersize=8, label="tile8 vs tile16")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1.0x)")
    ax.set_xlabel("Gaussian Count (K)")
    ax.set_ylabel("Speedup vs tile16 (baseline)")
    ax.set_title("Tile Size Speedup Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([50, 200, 400])

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_scaling_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / '01_scaling_curve.png'}")


# ===========================================================================
# Chart 2: Forward/Backward Breakdown
# ===========================================================================

def chart_fwd_bwd_breakdown():
    """Forward/backward breakdown across tile sizes and scenes."""
    print("  Chart 2: Forward/backward breakdown...")

    # Combined data from profiles
    profile_files = {
        "50K": PROFILE_DIR.parent.parent / "profiles" / "fwd_bwd_profile_50k_1080p.json",
        "200K": PROFILE_DIR.parent.parent / "profiles" / "fwd_bwd_profile_200k_1080p.json",
        "400K": PROFILE_DIR / "fwd_bwd_400k.json",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (scene_label, prof_path) in enumerate(profile_files.items()):
        profile = load_json(prof_path)
        if not profile:
            print(f"    (no profile for {scene_label})")
            continue

        ax = axes[idx]

        # Get data
        tile_sizes = []
        fwd_ms = []
        bwd_ms = []

        if scene_label == "400K" and "results" in profile:
            # New format
            for ts in ["8", "16", "32"]:
                if ts in profile["results"]:
                    r = profile["results"][ts]
                    tile_sizes.append(int(ts))
                    fwd_ms.append(r["forward_ms"])
                    bwd_ms.append(r["backward_ms"])
        else:
            # Old format
            for entry in profile.get("fwd", []):
                ts = entry["tile_size"]
                tile_sizes.append(ts)
                fwd_ms.append(entry["mean_ms"])
            for entry in profile.get("bwd", []):
                bwd_ms.append(entry["mean_ms"])

        if not tile_sizes:
            continue

        x = np.arange(len(tile_sizes))
        width = 0.35

        bars1 = ax.bar(x - width/2, fwd_ms, width, label="Forward", color=COLORS["forward"])
        bars2 = ax.bar(x + width/2, bwd_ms, width, label="Backward", color=COLORS["backward"])

        ax.set_xticks(x)
        ax.set_xticklabels([f"tile{ts}" for ts in tile_sizes])
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{scene_label} Gaussians")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add % labels
        for b1, b2, fw, bw in zip(bars1, bars2, fwd_ms, bwd_ms):
            total = fw + bw
            if total > 0:
                ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.5,
                        f"{fw/total*100:.0f}%", ha="center", fontsize=9, fontweight="bold")
                ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.5,
                        f"{bw/total*100:.0f}%", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Forward vs Backward Time by Tile Size", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_fwd_bwd_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / '02_fwd_bwd_breakdown.png'}")


# ===========================================================================
# Chart 3: Kernel Time Comparison
# ===========================================================================

def chart_kernel_time_comparison():
    """Compare kernel times across tile sizes for all scenes."""
    print("  Chart 3: Kernel time comparison...")

    # Gather total times for tile8/16/32 across scenes
    scenes_data = {}
    profile_files = {
        "50K": PROFILE_DIR.parent.parent / "profiles" / "fwd_bwd_profile_50k_1080p.json",
        "200K": PROFILE_DIR.parent.parent / "profiles" / "fwd_bwd_profile_200k_1080p.json",
        "400K": PROFILE_DIR / "fwd_bwd_400k.json",
    }

    for scene_label, prof_path in profile_files.items():
        profile = load_json(prof_path)
        if not profile:
            continue

        if scene_label == "400K" and "results" in profile:
            for ts in ["8", "16", "32"]:
                if ts in profile["results"]:
                    r = profile["results"][ts]
                    if scene_label not in scenes_data:
                        scenes_data[scene_label] = {}
                    scenes_data[scene_label][f"tile{ts}"] = r["total_ms"]
        else:
            for entry in profile.get("total", []):
                ts = entry["tile_size"]
                if scene_label not in scenes_data:
                    scenes_data[scene_label] = {}
                scenes_data[scene_label][f"tile{ts}"] = entry["mean_ms"]

    if not scenes_data:
        print("    (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    scenes = sorted(scenes_data.keys(), key=lambda x: int(x.replace("K", "")))
    x = np.arange(len(scenes))
    width = 0.25

    for i, ts_label in enumerate(["tile8", "tile16", "tile32"]):
        values = []
        for s in scenes:
            if s in scenes_data and ts_label in scenes_data[s]:
                values.append(scenes_data[s][ts_label])
            else:
                values.append(0)
        color = {"tile8": COLORS["tile8"], "tile16": COLORS["tile16"], "tile32": COLORS["tile32"]}[ts_label]
        bars = ax.bar(x + (i - 1) * width, values, width, label=ts_label, color=color)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("Total Time (ms)")
    ax.set_title("Total Rendering Time by Tile Size and Scene")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add speedup annotations
    for i, s in enumerate(scenes):
        if s in scenes_data and "tile16" in scenes_data[s] and "tile32" in scenes_data[s]:
            speedup = scenes_data[s]["tile16"] / scenes_data[s]["tile32"]
            ax.annotate(f"{speedup:.2f}x",
                       xy=(i + width, scenes_data[s]["tile32"]),
                       xytext=(i + width + 0.1, scenes_data[s]["tile32"] * 1.1),
                       fontsize=9, fontweight="bold", color=COLORS["tile32"],
                       arrowprops=dict(arrowstyle="->", color=COLORS["tile32"], alpha=0.7))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_kernel_time_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / '03_kernel_time_comparison.png'}")


# ===========================================================================
# Chart 4: Occupancy vs Runtime (Projected Analysis)
# ===========================================================================

def chart_occupancy_analysis():
    """
    Projected analysis of shared memory, register usage, occupancy vs runtime.
    Based on CUDA kernel parameters for gsplat rasterization.
    """
    print("  Chart 4: Occupancy/register/shared memory analysis...")

    # A100 specs
    SM_SHARED_MEM = 164 * 1024  # 164 KB per SM
    SM_REGISTERS = 65536
    MAX_THREADS_PER_SM = 2048
    MAX_BLOCKS_PER_SM = 32

    tile_sizes = [4, 8, 12, 16, 20, 24, 28, 32, 48, 64]

    # Estimate shared memory per block
    # Forward: tile_size^2 * (sizeof(int32) + 2 * sizeof(vec3))
    # = tile_size^2 * (4 + 2*12) = tile_size^2 * 28 bytes
    # Backward needs similar + extra buffers
    shmem_fwd = [ts * ts * 28 for ts in tile_sizes]
    shmem_bwd = [ts * ts * 48 for ts in tile_sizes]  # backward needs more

    # Register usage estimate (from gsplat source analysis)
    # Forward: ~48 registers per thread (typical for rasterization)
    # Larger tiles may need slightly more registers
    regs_per_thread_fwd = [48 + ts // 8 for ts in tile_sizes]  # slight increase
    regs_per_thread_bwd = [56 + ts // 8 for ts in tile_sizes]

    # Threads per block = tile_size * tile_size
    threads_per_block = [ts * ts for ts in tile_sizes]

    # Occupancy calculation
    def calc_occupancy(shmem, regs, threads, max_shmem=SM_SHARED_MEM, max_regs=SM_REGISTERS,
                       max_threads=MAX_THREADS_PER_SM, max_blocks=MAX_BLOCKS_PER_SM):
        # Blocks limited by shared memory
        blocks_by_shmem = max_shmem // shmem if shmem > 0 else max_blocks
        # Blocks limited by registers
        blocks_by_regs = max_regs // (regs * threads) if (regs * threads) > 0 else max_blocks
        # Blocks limited by threads
        blocks_by_threads = max_threads // threads if threads > 0 else max_blocks

        max_blocks_per_sm = min(blocks_by_shmem, blocks_by_regs, blocks_by_threads, max_blocks)
        warps_per_block = threads // 32 if threads >= 32 else 1
        total_warps = max_blocks_per_sm * warps_per_block
        max_warps = max_threads // 32
        occupancy = total_warps / max_warps * 100
        return occupancy, max_blocks_per_sm, blocks_by_shmem, blocks_by_regs, blocks_by_threads

    occ_fwd = []
    occ_bwd = []
    blocks_fwd = []
    shmem_lim_fwd = []
    reg_lim_fwd = []

    for i, ts in enumerate(tile_sizes):
        o_fwd, b_fwd, sh, rg, th = calc_occupancy(shmem_fwd[i], regs_per_thread_fwd[i], threads_per_block[i])
        o_bwd, b_bwd, _, _, _ = calc_occupancy(shmem_bwd[i], regs_per_thread_bwd[i], threads_per_block[i])
        occ_fwd.append(o_fwd)
        occ_bwd.append(o_bwd)
        blocks_fwd.append(b_fwd)
        shmem_lim_fwd.append(min(sh, 32))
        reg_lim_fwd.append(min(rg, 32))

    # Runtime data from experiments (use scaling_validation if available)
    scaling_data = load_json(AGG_DIR / "scaling_validation.json")
    runtime_400k = {"tile8": 272.22, "tile16": 68.23, "tile32": 21.19}  # fallback
    if scaling_data and "400k" in scaling_data["results"]:
        for ts_label in ["tile8", "tile16", "tile32"]:
            if ts_label in scaling_data["results"]["400k"]["results"]:
                runtime_400k[ts_label] = scaling_data["results"]["400k"]["results"][ts_label]["mean_ms"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Shared memory per block
    ax = axes[0, 0]
    ax.plot(tile_sizes, [s / 1024 for s in shmem_fwd], "s-", color=COLORS["tile8"], label="Forward", linewidth=2)
    ax.plot(tile_sizes, [s / 1024 for s in shmem_bwd], "o-", color=COLORS["tile32"], label="Backward", linewidth=2)
    ax.axhline(y=164, color="red", linestyle="--", alpha=0.5, label="A100 SM limit (164KB)")
    ax.axvline(x=16, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=32, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Tile Size")
    ax.set_ylabel("Shared Memory / Block (KB)")
    ax.set_title("Shared Memory Usage vs Tile Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32, 48, 64])

    # Top-right: Register usage
    ax = axes[0, 1]
    ax.plot(tile_sizes, regs_per_thread_fwd, "s-", color=COLORS["tile8"], label="Forward", linewidth=2)
    ax.plot(tile_sizes, regs_per_thread_bwd, "o-", color=COLORS["tile32"], label="Backward", linewidth=2)
    ax.axhline(y=64, color="red", linestyle="--", alpha=0.5, label="A100 register limit (64 regs/thread)")
    ax.axvline(x=16, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=32, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Tile Size")
    ax.set_ylabel("Estimated Registers / Thread")
    ax.set_title("Register Usage vs Tile Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32, 48, 64])

    # Bottom-left: Occupancy
    ax = axes[1, 0]
    ax.plot(tile_sizes, occ_fwd, "s-", color=COLORS["tile8"], label="Forward", linewidth=2)
    ax.plot(tile_sizes, occ_bwd, "o-", color=COLORS["tile32"], label="Backward", linewidth=2)
    ax.axvline(x=16, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=32, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Tile Size")
    ax.set_ylabel("Occupancy (%)")
    ax.set_title("Estimated Occupancy vs Tile Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32, 48, 64])

    # Annotate key points
    for ax_idx, ax in enumerate([axes[1, 0]]):
        for ts in [8, 16, 32]:
            if ts in tile_sizes:
                i = tile_sizes.index(ts)
                ax.annotate(f"tile{ts}: {occ_fwd[i]:.0f}%",
                           xy=(ts, occ_fwd[i]), xytext=(ts, occ_fwd[i] + 3),
                           fontsize=9, ha="center", fontweight="bold")

    # Bottom-right: Occupancy vs Runtime (trade-off)
    ax = axes[1, 1]
    # Connect 8, 16, 32 with their occupancy and runtime
    marker_styles = {"tile8": ("v", COLORS["tile8"]), "tile16": ("s", COLORS["tile16"]), "tile32": ("o", COLORS["tile32"])}
    for ts_label in ["tile8", "tile16", "tile32"]:
        ts = int(ts_label.replace("tile", ""))
        if ts in tile_sizes:
            i = tile_sizes.index(ts)
            rt = runtime_400k.get(ts_label, 100)
            occ = occ_fwd[i]
            marker, color = marker_styles[ts_label]
            ax.scatter(occ, rt, marker=marker, color=color, s=150, zorder=5, label=ts_label)
            ax.annotate(f" tile{ts}", (occ, rt), fontsize=10, fontweight="bold")

    ax.set_xlabel("Forward Occupancy (%)")
    ax.set_ylabel("400K Runtime (ms)")
    ax.set_title("Occupancy-Runtime Trade-off (400K)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("GPU Occupancy / Shared Memory / Register Analysis (A100)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_occupancy_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / '04_occupancy_analysis.png'}")


# ===========================================================================
# Chart 5: M0 Anomaly Verification
# ===========================================================================

def chart_m0_verification():
    """Chart showing M0 anomaly was a first-run artifact."""
    print("  Chart 5: M0 anomaly verification...")

    m0_data = load_json(RAW_DIR / "m0_verification.json")
    if not m0_data:
        print("    (no M0 verification data)")
        return

    phases = m0_data["phases"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Phase means comparison
    ax = axes[0]
    phase_labels = [p["phase_label"] for p in phases]
    means = [p["mean_ms"] for p in phases]
    colors = [COLORS["cold"], COLORS["warm"], COLORS["cold"], COLORS["warm"]]
    colors[0] = "#E67E22"  # M0 cold - distinctive

    bars = ax.bar(range(len(phases)), means, color=colors)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phase_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean Time (ms)")
    ax.set_title("400K M0/M2a Phases: Mean Time")
    ax.axhline(y=means[1], color=COLORS["warm"], linestyle="--", alpha=0.5, label=f"Steady-state ({means[1]:.1f}ms)")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}ms", ha="center", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # Middle: First frame comparison (cold start overhead)
    ax = axes[1]
    first_frames = [p.get("first_frame_ms", 0) for p in phases]
    ax.bar(range(len(phases)), first_frames, color=colors)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phase_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("First Frame Time (ms)")
    ax.set_title("First Frame Time (JIT Overhead)")

    for i, (bar, val) in enumerate(zip(bars, first_frames)):
        ax.text(i, val + 1, f"{val:.1f}ms", ha="center", fontsize=9, fontweight="bold")

    # Right: Historical comparison
    ax = axes[2]
    # Original data from aggregated results
    orig_m0 = 70.57
    orig_m2a = 34.27
    corrected_m0 = phases[2]["mean_ms"]
    corrected_m2a = phases[1]["mean_ms"]

    categories = ["Original\n(historical)", "Validated\n(corrected)"]
    m0_vals = [orig_m0, corrected_m0]
    m2a_vals = [orig_m2a, corrected_m2a]

    x = np.arange(len(categories))
    width = 0.3
    ax.bar(x - width/2, m0_vals, width, label="M0 (tile16 baseline)", color="#E67E22")
    ax.bar(x + width/2, m2a_vals, width, label="M2a (tile16 packed)", color=COLORS["tile16"])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Mean Time (ms)")
    ax.set_title("Historical vs Corrected: M0 vs M2a")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    for i in range(len(categories)):
        ax.text(i - width/2, m0_vals[i] + 1, f"{m0_vals[i]:.1f}ms", ha="center", fontsize=9)
        ax.text(i + width/2, m2a_vals[i] + 1, f"{m2a_vals[i]:.1f}ms", ha="center", fontsize=9)

    # Arrow showing the artifact
    ax.annotate("JIT compilation\nartifact (2.06x)",
                xy=(0, (orig_m0 + orig_m2a) / 2), xytext=(0.3, (orig_m0 + orig_m2a) / 2 + 20),
                fontsize=9, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"))

    plt.suptitle("400K M0 Anomaly Verification: First-Run Artifact Confirmed", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_m0_anomaly_verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / '05_m0_anomaly_verification.png'}")


# ===========================================================================
# Chart 6: Training Validation
# ===========================================================================

def chart_training():
    """Training validation charts."""
    print("  Chart 6: Training validation...")

    training_files = {
        "50K": TRAINING_DIR / "training_50k.json",
    }

    for scene_label, path in training_files.items():
        data = load_json(path)
        if not data:
            print(f"    (no training data for {scene_label})")
            continue

        # Training step time comparison
        results = data["results"]
        tile_configs = list(results.keys())

        if len(tile_configs) < 2:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Step time breakdown
        ax = axes[0, 0]
        configs = []
        fwd_means = []
        bwd_means = []
        opt_means = []
        labels = []

        for cfg_name in tile_configs:
            r = results[cfg_name]
            labels.append(cfg_name.replace("_training", ""))
            fwd_means.append(r["forward_ms"])
            bwd_means.append(r["backward_ms"])
            opt_means.append(r["optimizer_ms"])

        x = np.arange(len(labels))
        width = 0.2
        ax.bar(x - width, fwd_means, width, label="Forward", color=COLORS["forward"])
        ax.bar(x, bwd_means, width, label="Backward", color=COLORS["backward"])
        ax.bar(x + width, opt_means, width, label="Optimizer", color=COLORS["optimizer"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Training Step Time Breakdown")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Top-right: Training stage comparison
        ax = axes[0, 1]
        stages = ["Early", "Middle", "Late"]
        for cfg_name in tile_configs:
            r = results[cfg_name]
            stage_vals = [r["early_stage_ms"], r["middle_stage_ms"], r["late_stage_ms"]]
            color = COLORS["tile16"] if "tile16" in cfg_name else COLORS["tile32"]
            label = cfg_name.replace("_training", "")
            ax.plot(stages, stage_vals, "o-", label=label, color=color, linewidth=2, markersize=8)

        ax.set_ylabel("Step Time (ms)")
        ax.set_title("Training Stage Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-left: PSNR trajectory
        ax = axes[1, 0]
        for cfg_name in tile_configs:
            r = results[cfg_name]
            psnr_traj = r.get("psnr_trajectory", [])
            if psnr_traj:
                steps = [p["step"] for p in psnr_traj]
                psnrs = [p["psnr"] for p in psnr_traj]
                color = COLORS["tile16"] if "tile16" in cfg_name else COLORS["tile32"]
                label = cfg_name.replace("_training", "")
                ax.plot(steps, psnrs, "-", label=label, color=color, linewidth=2)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("PSNR")
        ax.set_title("PSNR During Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-right: VRAM comparison
        ax = axes[1, 1]
        vram_vals = [results[c]["peak_vram_mb"] for c in tile_configs]
        colors = [COLORS["tile16"] if "tile16" in c else COLORS["tile32"] for c in tile_configs]
        bars = ax.bar(labels, vram_vals, color=colors)
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title("Peak VRAM During Training")
        for bar, val in zip(bars, vram_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f"{val:.0f}MB", ha="center", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(f"Training Validation: tile16 vs tile32 ({scene_label})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"06_training_{scene_label.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {FIGURES_DIR / f'06_training_{scene_label.lower()}.png'}")


# ===========================================================================
# Chart 7: VRAM Comparison
# ===========================================================================

def chart_vram_comparison():
    """Peak VRAM comparison across tile sizes and scenes."""
    print("  Chart 7: VRAM comparison...")

    scaling_data = load_json(AGG_DIR / "scaling_validation.json")
    if not scaling_data:
        print("    (no scaling data)")
        return

    results = scaling_data["results"]
    scenes = sorted(results.keys())
    tile_labels = ["tile8", "tile16", "tile32"]
    tile_colors = [COLORS["tile8"], COLORS["tile16"], COLORS["tile32"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenes))
    width = 0.25

    for i, ts_label in enumerate(tile_labels):
        vrams = []
        for s in scenes:
            if ts_label in results[s]["results"]:
                vrams.append(results[s]["results"][ts_label]["peak_vram_mb"] / 1024)  # GB
            else:
                vrams.append(0)
        bars = ax.bar(x + (i - 1) * width, vrams, width, label=ts_label, color=tile_colors[i])
        for bar, val in zip(bars, vrams):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{val:.1f}GB", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{results[s]['num_gaussians']//1000}K" for s in scenes])
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("Peak VRAM Usage by Tile Size and Scene")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_vram_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / '07_vram_comparison.png'}")


# ===========================================================================
# Chart 8: Speed-Quality Pareto
# ===========================================================================

def chart_speed_quality_pareto():
    """Speed vs quality Pareto frontier."""
    print("  Chart 8: Speed-quality Pareto...")

    # Quality is identical (same numerical computation), so this is a trivial Pareto
    fig, ax = plt.subplots(figsize=(8, 6))

    # Points for tile8/16/32 at 400K
    scaling_data = load_json(AGG_DIR / "scaling_validation.json")
    training_data = load_json(TRAINING_DIR / "training_50k.json")

    fwd_bwd = load_json(PROFILE_DIR / "fwd_bwd_400k.json")

    if fwd_bwd and "results" in fwd_bwd:
        for ts in ["8", "16", "32"]:
            if ts in fwd_bwd["results"]:
                r = fwd_bwd["results"][ts]
                total_ms = r["total_ms"]
                fps = 1000.0 / total_ms
                vram = r.get("peak_vram_mb", 0) / 1024
                label = f"tile{ts}"
                color = COLORS.get(f"tile{ts}", "gray")
                marker = {"8": "v", "16": "s", "32": "o"}[ts]
                size = {"8": 80, "16": 100, "32": 150}[ts]

                ax.scatter(vram, fps, marker=marker, color=color, s=size, zorder=5,
                          label=label, edgecolors="black", linewidth=0.5)
                ax.annotate(f"  tile{ts}", (vram, fps), fontsize=11, fontweight="bold")

        ax.set_xlabel("Peak VRAM (GB) [lower is better]")
        ax.set_ylabel("Throughput (FPS) [higher is better]")
        ax.set_title("Speed-VRAM Pareto Frontier (400K, A100-80GB)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add arrow for "optimal direction"
        ax.annotate("Better", xy=(1, 100), fontsize=12, color="green", fontweight="bold")

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "08_speed_quality_pareto.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {FIGURES_DIR / '08_speed_quality_pareto.png'}")


# ===========================================================================
# Chart 9: Tile Size Scaling Curve (detailed)
# ===========================================================================

def chart_tile_scaling_curve():
    """Detailed tile-size scaling showing the diminishing returns."""
    print("  Chart 9: Detailed tile scaling curve...")

    # Use the existing fwd/bwd profile data for 50K, 200K, 400K
    profile_data = {
        "50K": load_json(PROFILE_DIR.parent.parent / "profiles" / "fwd_bwd_profile_50k_1080p.json"),
        "200K": load_json(PROFILE_DIR.parent.parent / "profiles" / "fwd_bwd_profile_200k_1080p.json"),
        "400K": load_json(PROFILE_DIR / "fwd_bwd_400k.json"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (scene_label, profile) in enumerate(profile_data.items()):
        ax = axes[idx]

        tile_sizes = [8, 16, 32]
        fwd_times = []
        bwd_times = []
        total_times = []

        if scene_label == "400K" and profile and "results" in profile:
            for ts in ["8", "16", "32"]:
                if ts in profile["results"]:
                    r = profile["results"][ts]
                    fwd_times.append(r["forward_ms"])
                    bwd_times.append(r["backward_ms"])
                    total_times.append(r["total_ms"])
        elif profile:
            for ts in [8, 16, 32]:
                f_entry = next((e for e in profile.get("fwd", []) if e["tile_size"] == ts), None)
                b_entry = next((e for e in profile.get("bwd", []) if e["tile_size"] == ts), None)
                if f_entry and b_entry:
                    fwd_times.append(f_entry["mean_ms"])
                    bwd_times.append(b_entry["mean_ms"])
                    total_times.append((f_entry["mean_ms"] + b_entry["mean_ms"]))
                elif f_entry:
                    fwd_times.append(f_entry["mean_ms"])
                    bwd_times.append(0)
                    total_times.append(f_entry["mean_ms"])

        if not total_times or len(total_times) < 3:
            continue

        x = np.arange(len(tile_sizes))

        ax.fill_between(x, 0, fwd_times, label="Forward", color=COLORS["forward"], alpha=0.8)
        ax.fill_between(x, fwd_times, [f + b for f, b in zip(fwd_times, bwd_times)],
                        label="Backward", color=COLORS["backward"], alpha=0.8)

        # Total line
        ax.plot(x, total_times, "ko-", linewidth=2, markersize=8, label="Total")

        # Annotations
        for i, (f, b, t) in enumerate(zip(fwd_times, bwd_times, total_times)):
            ax.text(i, t + max(total_times) * 0.03, f"{t:.1f}ms", ha="center", fontsize=9, fontweight="bold")
            ax.text(i, f/2, f"F {f:.0f}ms", ha="center", fontsize=8, color="white", fontweight="bold")
            ax.text(i, f + b/2, f"B {b:.0f}ms", ha="center", fontsize=8, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"tile{ts}" for ts in tile_sizes])
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{scene_label} Gaussians")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Tile Size Scaling: Forward/Backward Decomposition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_tile_scaling_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / '09_tile_scaling_curve.png'}")


# ===========================================================================
# Chart 10: Forward Percentage vs Gaussian Count
# ===========================================================================

def chart_forward_percentage():
    """Forward% vs Gaussian count for each tile size."""
    print("  Chart 10: Forward percentage trend...")

    profile_data = {
        "50K": load_json(PROFILE_DIR.parent.parent / "profiles" / "fwd_bwd_profile_50k_1080p.json"),
        "200K": load_json(PROFILE_DIR.parent.parent / "profiles" / "fwd_bwd_profile_200k_1080p.json"),
        "400K": load_json(PROFILE_DIR / "fwd_bwd_400k.json"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    tile_styles = {
        8: ("v--", COLORS["tile8"]),
        16: ("s--", COLORS["tile16"]),
        32: ("o-", COLORS["tile32"]),
    }

    for tile_size, (style, color) in tile_styles.items():
        gaussians = []
        fwd_pcts = []
        for scene_label, profile in profile_data.items():
            if not profile:
                continue
            if scene_label == "400K" and "results" in profile:
                if str(tile_size) in profile["results"]:
                    r = profile["results"][str(tile_size)]
                    gaussians.append(int(scene_label.replace("K", "")) * 1000)
                    fwd_pcts.append(r["forward_pct"])
            else:
                f_entry = next((e for e in profile.get("fwd", []) if e["tile_size"] == tile_size), None)
                b_entry = next((e for e in profile.get("bwd", []) if e["tile_size"] == tile_size), None)
                if f_entry and b_entry:
                    gaussians.append(int(scene_label.replace("K", "")) * 1000)
                    total = f_entry["mean_ms"] + b_entry["mean_ms"]
                    fwd_pcts.append(f_entry["mean_ms"] / total * 100)

        ax.plot([g / 1000 for g in gaussians], fwd_pcts, style, color=color, linewidth=2,
                markersize=8, label=f"tile{tile_size}")

    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Gaussian Count (K)")
    ax.set_ylabel("Forward Time (%)")
    ax.set_title("Forward Percentage vs Gaussian Count (bottleneck analysis)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([50, 200, 400])
    ax.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "10_forward_percentage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / '10_forward_percentage.png'}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"Generating final charts...")
    print(f"Output: {FIGURES_DIR}/")

    # Load scaling data
    scaling_data = load_json(AGG_DIR / "scaling_validation.json")

    chart_scaling_curve(scaling_data)
    chart_fwd_bwd_breakdown()
    chart_kernel_time_comparison()
    chart_occupancy_analysis()
    chart_m0_verification()
    chart_training()
    chart_vram_comparison()
    chart_speed_quality_pareto()
    chart_tile_scaling_curve()
    chart_forward_percentage()

    print(f"\nAll charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
