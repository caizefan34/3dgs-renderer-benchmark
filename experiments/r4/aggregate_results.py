#!/usr/bin/env python3
"""
R4 13-Scene Final Aggregation Script (v2)

Collects training metrics from all 13 scenes (baseline + candidate_c),
computes PSNR/SSIM deltas, timing comparison, and generates the final report.

Handles multiple metric file formats:
1. training_metrics.json (reference_v1 trainer output)
2. quality_baseline.json (s22 format)
3. Log file parsing for in-progress training

Usage:
  python3 experiments/r4/aggregate_results.py
"""
import json, os, sys, re
from pathlib import Path

REPO_ROOT = Path(os.path.expanduser("~/3dgs-renderer-benchmark"))
OUTPUT_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")
BASELINE_DIR = REPO_ROOT / "results" / "reference_v1"
LOG_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs")

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


def parse_log_for_eval(log_path):
    """Parse training log for PSNR/SSIM evaluations."""
    if not os.path.exists(log_path):
        return {}
    
    results = {}
    current_iter = None
    
    with open(log_path) as f:
        for line in f:
            # Match ITER lines
            m = re.search(r'\[ITER (\d+)\]', line)
            if m:
                current_iter = int(m.group(1))
            
            # Match PSNR lines
            m = re.search(r'PSNR=([\d.]+)\s+SSIM=([\d.]+)\s+L1=([\d.]+)\s+N=(\d+)', line)
            if m and current_iter is not None:
                results[current_iter] = {
                    "psnr": float(m.group(1)),
                    "ssim": float(m.group(2)),
                    "l1": float(m.group(3)),
                    "N_gaussians": int(m.group(4)),
                    "iteration": current_iter,
                }
    
    return results


def parse_log_for_timing(log_path):
    """Parse training log for timing info from ITER lines."""
    if not os.path.exists(log_path):
        return None
    
    times = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'fwd=([\d.]+)ms\s+bwd=([\d.]+)ms\s+total=([\d.]+)ms', line)
            if m:
                times.append({
                    "fwd": float(m.group(1)),
                    "bwd": float(m.group(2)),
                    "total": float(m.group(3)),
                })
    
    if not times:
        return None
    
    n = len(times)
    return {
        "fwd_ms_mean": sum(t["fwd"] for t in times) / n,
        "bwd_ms_mean": sum(t["bwd"] for t in times) / n,
        "total_ms_mean": sum(t["total"] for t in times) / n,
        "n_samples": n,
    }


def get_final_eval(checkpoints):
    """Get the final (highest iteration) evaluation."""
    if not checkpoints:
        return None
    max_iter = max(checkpoints.keys(), key=int)
    return checkpoints[max_iter]


def find_baseline_metrics(scene):
    """Find baseline training metrics from multiple sources."""
    # 1. Check r4_13scene_v2 baseline
    path = OUTPUT_BASE / scene / "baseline" / "training_metrics.json"
    if path.exists():
        d = json.load(open(path))
        timing = None
        timing_path = path.parent / "timing.json"
        if timing_path.exists():
            t = json.load(open(timing_path))
            timing = t.get("full_run", t)
        return d.get("checkpoints", {}), timing
    
    # 2. Check log file for in-progress training
    log_path = LOG_BASE / f"{scene}_baseline.log"
    if log_path.exists():
        ckpts = parse_log_for_eval(log_path)
        timing = parse_log_for_timing(log_path)
        return ckpts, timing
    
    # 3. Check existing reference_v1 baselines
    if scene == "room":
        path = BASELINE_DIR / "room_30k" / "training_metrics.json"
        if path.exists():
            d = json.load(open(path))
            timing = None
            timing_path = BASELINE_DIR / "room_30k" / "timing.json"
            if timing_path.exists():
                t = json.load(open(timing_path))
                timing = t.get("full_run", t)
            return d.get("checkpoints", {}), timing
    
    for s in ["s22"]:
        # Check quality_baseline.json
        qb_path = BASELINE_DIR / s / scene / "quality_baseline.json"
        if qb_path.exists():
            d = json.load(open(qb_path))
            return d.get("checkpoints", {}), None
        # Check training_metrics.json
        tm_path = BASELINE_DIR / s / scene / "training_metrics.json"
        if tm_path.exists():
            d = json.load(open(tm_path))
            return d.get("checkpoints", {}), None
    
    return {}, None


def find_candidate_metrics(scene):
    """Find candidate_c training metrics."""
    # 1. Check training_metrics.json (saved at end)
    path = OUTPUT_BASE / scene / "candidate_c" / "training_metrics.json"
    if path.exists():
        d = json.load(open(path))
        timing = None
        timing_path = path.parent / "timing.json"
        if timing_path.exists():
            t = json.load(open(timing_path))
            timing = t.get("full_run", t)
        return d.get("checkpoints", {}), timing
    
    # 2. Check log files for in-progress training
    for log_path in [
        Path(os.path.expanduser(f"~/r4_{scene}_candidate.log")),
        LOG_BASE / f"{scene}_candidate.log",
    ]:
        if log_path.exists():
            ckpts = parse_log_for_eval(log_path)
            timing = parse_log_for_timing(log_path)
            if ckpts:
                return ckpts, timing
    
    return {}, None


def main():
    print("=" * 130)
    print("R4: 13-Scene Final Aggregation")
    print("=" * 130)
    
    results = []
    
    for scene in SCENES:
        dtype = DATASET_TYPE[scene]
        
        b_ckpts, b_timing = find_baseline_metrics(scene)
        c_ckpts, c_timing = find_candidate_metrics(scene)
        
        b_eval = get_final_eval(b_ckpts)
        c_eval = get_final_eval(c_ckpts)
        
        # Also get 30K if available, otherwise highest
        b_30k = b_ckpts.get("30000", b_eval)
        c_30k = c_ckpts.get("30000", c_eval)
        
        # Only compare when BOTH have 30K data
        has_30k = "30000" in b_ckpts and "30000" in c_ckpts
        
        row = {
            "scene": scene,
            "dataset": dtype,
            "baseline_psnr": b_30k["psnr"] if b_30k else None,
            "baseline_ssim": b_30k["ssim"] if b_30k else None,
            "baseline_n": b_30k["N_gaussians"] if b_30k else None,
            "baseline_iter": b_30k["iteration"] if b_30k else None,
            "baseline_time_ms": b_timing["total_ms_mean"] if b_timing else None,
            "baseline_fwd_ms": b_timing["fwd_ms_mean"] if b_timing else None,
            "baseline_bwd_ms": b_timing["bwd_ms_mean"] if b_timing else None,
            "candidate_psnr": c_30k["psnr"] if c_30k else None,
            "candidate_ssim": c_30k["ssim"] if c_30k else None,
            "candidate_n": c_30k["N_gaussians"] if c_30k else None,
            "candidate_iter": c_30k["iteration"] if c_30k else None,
            "candidate_time_ms": c_timing["total_ms_mean"] if c_timing else None,
            "candidate_fwd_ms": c_timing["fwd_ms_mean"] if c_timing else None,
            "candidate_bwd_ms": c_timing["bwd_ms_mean"] if c_timing else None,
            "baseline_ckpts": len(b_ckpts),
            "candidate_ckpts": len(c_ckpts),
        }
        
        if row["baseline_psnr"] and row["candidate_psnr"] and has_30k:
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
    print(f"\n{'Scene':<12} {'Dataset':<14} {'B_PSNR':>7} {'C_PSNR':>7} {'ΔPSNR':>7} {'B_SSIM':>7} {'C_SSIM':>7} {'ΔSSIM':>7} {'B_ms':>7} {'C_ms':>7} {'Speed':>6} {'B_N':>8} {'C_N':>8} {'B_it':>6} {'C_it':>6}")
    print("-" * 140)
    
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
        b_it = f"{r['baseline_iter']}" if r['baseline_iter'] else "  N/A"
        c_it = f"{r['candidate_iter']}" if r['candidate_iter'] else "  N/A"
        
        print(f"{r['scene']:<12} {r['dataset']:<14} {b_psnr:>7} {c_psnr:>7} {d_psnr:>7} {b_ssim:>7} {c_ssim:>7} {d_ssim:>7} {b_ms:>7} {c_ms:>7} {spd:>6} {b_n:>8} {c_n:>8} {b_it:>6} {c_it:>6}")
    
    # Summary
    complete = [r for r in results if r['psnr_delta'] is not None]
    print(f"\n=== Summary ===")
    print(f"Complete (both at 30K): {len(complete)}/{len(SCENES)} scenes")
    print(f"Baseline 30K data: {sum(1 for r in results if r['baseline_iter'] == 30000)}/13")
    print(f"Candidate 30K data: {sum(1 for r in results if r['candidate_iter'] == 30000)}/13")
    print(f"Candidate in progress: {sum(1 for r in results if r['candidate_iter'] and r['candidate_iter'] < 30000)}/13")
    
    if complete:
        avg_psnr_delta = sum(r['psnr_delta'] for r in complete) / len(complete)
        avg_ssim_delta = sum(r['ssim_delta'] for r in complete) / len(complete)
        psnr_positive = sum(1 for r in complete if r['psnr_delta'] >= 0)
        psnr_minor_drop = sum(1 for r in complete if -0.5 <= r['psnr_delta'] < 0)
        psnr_significant_drop = sum(1 for r in complete if r['psnr_delta'] < -0.5)
        
        print(f"\nAverage PSNR delta: {avg_psnr_delta:+.2f} dB")
        print(f"Average SSIM delta: {avg_ssim_delta:+.4f}")
        print(f"PSNR >= baseline: {psnr_positive}")
        print(f"PSNR minor drop (< 0.5 dB): {psnr_minor_drop}")
        print(f"PSNR significant drop (>= 0.5 dB): {psnr_significant_drop}")
        
        speedups = [r['speedup'] for r in complete if r['speedup']]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"\nAverage speedup: {avg_speedup:.2f}x")
    
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
        else:
            print("MIXED: Some scenes show significant quality degradation")
            print("  Failed scenes preserved as research evidence")
    else:
        print(f"INCOMPLETE: {len(complete)}/{len(SCENES)} scenes finished")


if __name__ == "__main__":
    main()
