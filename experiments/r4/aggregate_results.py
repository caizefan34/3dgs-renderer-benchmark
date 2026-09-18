#!/usr/bin/env python3
"""
R4 13-Scene Final Aggregation Script

Collects training metrics from all 13 scenes (baseline + candidate_c),
computes PSNR/SSIM deltas, timing comparison, and generates the final report.

Usage:
  python3 experiments/r4/aggregate_results.py
"""
import json, os, sys, glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")
BASELINE_DIR = REPO_ROOT / "results" / "reference_v1"

SCENES = [
    "bicycle", "bonsai", "counter", "flowers", "garden", "kitchen",
    "room", "stump", "treehill", "train", "truck", "drjohnson", "playroom"
]

DATASET_TYPE = {
    "bicycle": "mipnerf360", "bonsai": "mipnerf360", "counter": "mipnerf360",
    "flowers": "mipnerf360", "garden": "mipnerf360", "kitchen": "mipnerf360",
    "room": "mipnerf360", "stump": "mipnerf360", "treehill": "mipnerf360",
    "train": "tanksandtemples", "truck": "tanksandtemples",
    "drjohnson": "deepblending", "playroom": "deepblending",
}


def load_metrics(path):
    """Load training_metrics.json."""
    try:
        return json.load(open(path))
    except:
        return None


def load_timing(path):
    """Load timing.json."""
    try:
        return json.load(open(path))
    except:
        return None


def get_final_eval(metrics):
    """Get the final (30K) evaluation from metrics."""
    if not metrics or "checkpoints" not in metrics:
        return None
    ckpts = metrics["checkpoints"]
    if "30000" in ckpts:
        return ckpts["30000"]
    # Get the highest iteration
    if ckpts:
        max_iter = max(ckpts.keys(), key=int)
        return ckpts[max_iter]
    return None


def get_timing_stats(timing):
    """Get full_run timing stats."""
    if not timing:
        return None
    if "full_run" in timing:
        return timing["full_run"]
    # Get any non-zero entry
    for k, v in timing.items():
        if v.get("total_ms_mean", 0) > 0:
            return v
    return None


def find_baseline_metrics(scene):
    """Find baseline training metrics."""
    # Check r4_13scene_v2 baseline
    path = OUTPUT_BASE / scene / "baseline" / "training_metrics.json"
    if path.exists():
        return load_metrics(path), load_timing(str(path.parent / "timing.json"))

    # Check existing reference_v1 baselines
    if scene == "room":
        path = BASELINE_DIR / "room_30k" / "training_metrics.json"
        if path.exists():
            return load_metrics(path), load_timing(str(path.parent / "timing.json"))
    
    for s in ["s22"]:
        path = BASELINE_DIR / s / scene / "training_metrics.json"
        if path.exists():
            return load_metrics(path), load_timing(str(path.parent / "timing.json"))
    
    return None, None


def find_candidate_metrics(scene):
    """Find candidate_c training metrics."""
    path = OUTPUT_BASE / scene / "candidate_c" / "training_metrics.json"
    if path.exists():
        return load_metrics(path), load_timing(str(path.parent / "timing.json"))
    return None, None


def main():
    print("=" * 120)
    print("R4: 13-Scene Final Aggregation")
    print("=" * 120)
    
    results = []
    
    for scene in SCENES:
        dtype = DATASET_TYPE[scene]
        
        baseline_metrics, baseline_timing = find_baseline_metrics(scene)
        candidate_metrics, candidate_timing = find_candidate_metrics(scene)
        
        b_eval = get_final_eval(baseline_metrics)
        c_eval = get_final_eval(candidate_metrics)
        b_time = get_timing_stats(baseline_timing)
        c_time = get_timing_stats(candidate_timing)
        
        row = {
            "scene": scene,
            "dataset": dtype,
            "baseline_psnr": b_eval["psnr"] if b_eval else None,
            "baseline_ssim": b_eval["ssim"] if b_eval else None,
            "baseline_n": b_eval["N_gaussians"] if b_eval else None,
            "baseline_time_ms": b_time["total_ms_mean"] if b_time else None,
            "baseline_fwd_ms": b_time["fwd_ms_mean"] if b_time else None,
            "baseline_bwd_ms": b_time["bwd_ms_mean"] if b_time else None,
            "candidate_psnr": c_eval["psnr"] if c_eval else None,
            "candidate_ssim": c_eval["ssim"] if c_eval else None,
            "candidate_n": c_eval["N_gaussians"] if c_eval else None,
            "candidate_time_ms": c_time["total_ms_mean"] if c_time else None,
            "candidate_fwd_ms": c_time["fwd_ms_mean"] if c_time else None,
            "candidate_bwd_ms": c_time["bwd_ms_mean"] if c_time else None,
        }
        
        if row["baseline_psnr"] and row["candidate_psnr"]:
            row["psnr_delta"] = row["candidate_psnr"] - row["baseline_psnr"]
            row["ssim_delta"] = row["candidate_ssim"] - row["baseline_ssim"]
            if row["baseline_time_ms"] and row["candidate_time_ms"]:
                row["speedup"] = row["baseline_time_ms"] / row["candidate_time_ms"]
            else:
                row["speedup"] = None
        else:
            row["psnr_delta"] = None
            row["ssim_delta"] = None
            row["speedup"] = None
        
        results.append(row)
    
    # Print table
    print(f"\n{'Scene':<12} {'Dataset':<14} {'B_PSNR':>7} {'C_PSNR':>7} {'ΔPSNR':>7} {'B_SSIM':>7} {'C_SSIM':>7} {'ΔSSIM':>7} {'B_ms':>7} {'C_ms':>7} {'Speed':>6} {'B_N':>8} {'C_N':>8}")
    print("-" * 120)
    
    for r in results:
        b_psnr = f"{r['baseline_psnr']:.2f}" if r['baseline_psnr'] else "  N/A"
        c_psnr = f"{r['candidate_psnr']:.2f}" if r['candidate_psnr'] else "  N/A"
        d_psnr = f"{r['psnr_delta']:+.2f}" if r['psnr_delta'] is not None else "  N/A"
        b_ssim = f"{r['baseline_ssim']:.4f}" if r['baseline_ssim'] else "  N/A"
        c_ssim = f"{r['candidate_ssim']:.4f}" if r['candidate_ssim'] else "  N/A"
        d_ssim = f"{r['ssim_delta']:+.4f}" if r['ssim_delta'] is not None else "  N/A"
        b_ms = f"{r['baseline_time_ms']:.1f}" if r['baseline_time_ms'] else "  N/A"
        c_ms = f"{r['candidate_time_ms']:.1f}" if r['candidate_time_ms'] else "  N/A"
        spd = f"{r['speedup']:.2f}x" if r['speedup'] else "  N/A"
        b_n = f"{r['baseline_n']}" if r['baseline_n'] else "  N/A"
        c_n = f"{r['candidate_n']}" if r['candidate_n'] else "  N/A"
        
        print(f"{r['scene']:<12} {r['dataset']:<14} {b_psnr:>7} {c_psnr:>7} {d_psnr:>7} {b_ssim:>7} {c_ssim:>7} {d_ssim:>7} {b_ms:>7} {c_ms:>7} {spd:>6} {b_n:>8} {c_n:>8}")
    
    # Summary
    complete = [r for r in results if r['psnr_delta'] is not None]
    print(f"\n=== Summary ===")
    print(f"Complete: {len(complete)}/{len(SCENES)} scenes")
    
    if complete:
        avg_psnr_delta = sum(r['psnr_delta'] for r in complete) / len(complete)
        avg_ssim_delta = sum(r['ssim_delta'] for r in complete) / len(complete)
        psnr_positive = sum(1 for r in complete if r['psnr_delta'] >= 0)
        psnr_minor_drop = sum(1 for r in complete if -0.5 <= r['psnr_delta'] < 0)
        psnr_significant_drop = sum(1 for r in complete if r['psnr_delta'] < -0.5)
        
        print(f"Average PSNR delta: {avg_psnr_delta:+.2f} dB")
        print(f"Average SSIM delta: {avg_ssim_delta:+.4f}")
        print(f"PSNR >= baseline: {psnr_positive}")
        print(f"PSNR minor drop (< 0.5 dB): {psnr_minor_drop}")
        print(f"PSNR significant drop (>= 0.5 dB): {psnr_significant_drop}")
        
        speedups = [r['speedup'] for r in complete if r['speedup']]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"Average speedup: {avg_speedup:.2f}x")
    
    # Save JSON
    output_path = OUTPUT_BASE / "final_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Verdict
    print(f"\n=== Verdict ===")
    if len(complete) == len(SCENES):
        if avg_psnr_delta > -0.5 and psnr_significant_drop <= 3:
            print("PASS: Candidate C maintains quality within 0.5 dB average")
            print(f"  Average PSNR drop: {avg_psnr_delta:+.2f} dB")
            print(f"  Scenes with significant drop: {psnr_significant_drop}/13")
        else:
            print("MIXED: Some scenes show significant quality degradation")
            print(f"  Average PSNR drop: {avg_psnr_delta:+.2f} dB")
            print(f"  Scenes with significant drop: {psnr_significant_drop}/13")
            print("  Failed scenes preserved as research evidence")
    else:
        print(f"INCOMPLETE: {len(complete)}/{len(SCENES)} scenes finished")


if __name__ == "__main__":
    main()
