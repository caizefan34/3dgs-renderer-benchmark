#!/usr/bin/env python3
"""
R5-A: Multi-seed Results Aggregation

Collects results from all 12 paired runs (2 scenes × 3 seeds × 2 methods),
computes paired deltas, and generates statistical analysis.

Seed 0 results are read from R4 paths.
Seeds 1-2 results are read from R5-A paths.
"""
import json, os, re, sys
from pathlib import Path
import numpy as np

# Paths
R4_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")
R5_BASE = Path("/mnt/storage_pool/liaoyuanjun/r5_a")
R5_LOGS = Path("/mnt/storage_pool/liaoyuanjun/r5_a_logs")

SCENES = ["train", "truck"]
SEEDS = [0, 1, 2]
METHODS = ["baseline", "candidate_c"]


def load_metrics_json(path):
    """Load training_metrics.json."""
    try:
        return json.load(open(path))
    except:
        return None


def parse_log_for_eval(log_path):
    """Parse training log for PSNR/SSIM evaluations."""
    if not os.path.exists(log_path):
        return {}
    results = {}
    current_iter = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[ITER (\d+)\]', line)
            if m:
                current_iter = int(m.group(1))
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


def get_run_data(scene, seed, method):
    """Get all checkpoint metrics for a run."""
    # Determine path
    if seed == 0:
        run_dir = R4_BASE / scene / method
    else:
        run_dir = R5_BASE / f"{scene}_seed{seed}" / method
    
    # Try training_metrics.json first
    metrics_path = run_dir / "training_metrics.json"
    if metrics_path.exists():
        d = json.load(open(metrics_path))
        ckpts = d.get("checkpoints", {})
        # Also get timing
        timing = None
        timing_path = run_dir / "timing.json"
        if timing_path.exists():
            t = json.load(open(timing_path))
            timing = t.get("full_run", t)
        return ckpts, timing, str(run_dir)
    
    # Fall back to log parsing
    if seed == 0:
        log_path = R5_LOGS / f"{scene}_seed{seed}_{method}.log"
    else:
        log_path = R5_LOGS / f"{scene}_seed{seed}_{method}.log"
    
    # Also check R4 log paths
    if seed == 0 and not log_path.exists():
        log_path = Path(f"/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/{scene}_{method}.log")
    
    ckpts = parse_log_for_eval(log_path)
    return ckpts, None, str(run_dir)


def main():
    print("=" * 100)
    print("R5-A: T&T Multi-Seed Reproducibility — Results Aggregation")
    print("=" * 100)
    
    all_results = []
    
    for scene in SCENES:
        for seed in SEEDS:
            for method in METHODS:
                ckpts, timing, run_dir = get_run_data(scene, seed, method)
                
                final = ckpts.get("30000") if ckpts else None
                if not final and ckpts:
                    max_iter = max(ckpts.keys(), key=int)
                    final = ckpts[max_iter]
                
                row = {
                    "scene": scene,
                    "seed": seed,
                    "method": method,
                    "psnr": final["psnr"] if final else None,
                    "ssim": final["ssim"] if final else None,
                    "n_gaussians": final["N_gaussians"] if final else None,
                    "iteration": final["iteration"] if final else None,
                    "training_time": None,  # TODO: from timing
                    "run_dir": run_dir,
                    "checkpoints": ckpts,
                }
                
                if timing:
                    row["training_time"] = timing.get("total_ms_mean")
                
                all_results.append(row)
    
    # Print table
    print(f"\n{'Scene':<8} {'Seed':>4} {'Method':<12} {'PSNR':>8} {'SSIM':>8} {'N_GS':>8} {'Iter':>6}")
    print("-" * 60)
    
    for r in all_results:
        psnr = f"{r['psnr']:.2f}" if r['psnr'] else "N/A"
        ssim = f"{r['ssim']:.4f}" if r['ssim'] else "N/A"
        ngs = f"{r['n_gaussians']}" if r['n_gaussians'] else "N/A"
        it = f"{r['iteration']}" if r['iteration'] else "N/A"
        print(f"{r['scene']:<8} {r['seed']:>4} {r['method']:<12} {psnr:>8} {ssim:>8} {ngs:>8} {it:>6}")
    
    # Compute paired deltas
    print(f"\n{'='*100}")
    print("Paired Deltas (ΔPSNR = C - B)")
    print(f"{'='*100}")
    
    paired_results = []
    for scene in SCENES:
        print(f"\n--- {scene} ---")
        for seed in SEEDS:
            b = next((r for r in all_results if r["scene"] == scene and r["seed"] == seed and r["method"] == "baseline"), None)
            c = next((r for r in all_results if r["scene"] == scene and r["seed"] == seed and r["method"] == "candidate_c"), None)
            
            if b and c and b["psnr"] and c["psnr"]:
                d_psnr = c["psnr"] - b["psnr"]
                d_ssim = c["ssim"] - b["ssim"]
                d_ngs = c["n_gaussians"] - b["n_gaussians"] if b["n_gaussians"] and c["n_gaussians"] else None
                r_gs = c["n_gaussians"] / b["n_gaussians"] if b["n_gaussians"] and c["n_gaussians"] else None
                
                paired_results.append({
                    "scene": scene,
                    "seed": seed,
                    "b_psnr": b["psnr"],
                    "c_psnr": c["psnr"],
                    "delta_psnr": d_psnr,
                    "b_ssim": b["ssim"],
                    "c_ssim": c["ssim"],
                    "delta_ssim": d_ssim,
                    "b_ngs": b["n_gaussians"],
                    "c_ngs": c["n_gaussians"],
                    "delta_ngs": d_ngs,
                    "r_gs": r_gs,
                    "b_iter": b["iteration"],
                    "c_iter": c["iteration"],
                })
                
                pos = "+" if d_psnr > 0 else ""
                print(f"  seed={seed}: B={b['psnr']:.2f} C={c['psnr']:.2f} ΔPSNR={pos}{d_psnr:.2f} ΔSSIM={d_ssim:+.4f} R_GS={r_gs:.3f}" if r_gs else f"  seed={seed}: B={b['psnr']:.2f} C={c['psnr']:.2f} ΔPSNR={pos}{d_psnr:.2f}")
            else:
                status = "INCOMPLETE"
                if b and not b["psnr"]:
                    status += " (B missing)"
                if c and not c["psnr"]:
                    status += " (C missing)"
                if not b:
                    status += " (B not found)"
                if not c:
                    status += " (C not found)"
                print(f"  seed={seed}: {status}")
    
    # Per-scene statistics
    print(f"\n{'='*100}")
    print("Per-Scene Statistics")
    print(f"{'='*100}")
    
    for scene in SCENES:
        scene_paired = [p for p in paired_results if p["scene"] == scene]
        if len(scene_paired) < 1:
            print(f"\n{scene}: insufficient data")
            continue
        
        deltas = [p["delta_psnr"] for p in scene_paired]
        ssim_deltas = [p["delta_ssim"] for p in scene_paired]
        r_gs_vals = [p["r_gs"] for p in scene_paired if p["r_gs"]]
        n_positive = sum(1 for d in deltas if d > 0)
        
        print(f"\n--- {scene} ({len(scene_paired)} seeds) ---")
        print(f"  Paired ΔPSNR values: {['%+.2f' % d for d in deltas]}")
        print(f"  Mean ΔPSNR: {np.mean(deltas):+.2f} dB")
        if len(deltas) >= 2:
            print(f"  Median ΔPSNR: {np.median(deltas):+.2f} dB")
            print(f"  Std ΔPSNR: {np.std(deltas, ddof=1):.2f} dB")
        print(f"  Min/Max ΔPSNR: {min(deltas):+.2f} / {max(deltas):+.2f}")
        print(f"  Positive seeds: {n_positive}/{len(deltas)}")
        print(f"  Mean ΔSSIM: {np.mean(ssim_deltas):+.4f}")
        if r_gs_vals:
            print(f"  Mean R_GS (C/B): {np.mean(r_gs_vals):.3f}")
        
        # 95% CI for paired mean (t-distribution)
        if len(deltas) >= 2:
            from scipy import stats as scipy_stats
            mean = np.mean(deltas)
            sem = np.std(deltas, ddof=1) / np.sqrt(len(deltas))
            if len(deltas) == 2:
                t_crit = 12.706  # df=1, 95%
            elif len(deltas) == 3:
                t_crit = 4.303   # df=2, 95%
            else:
                t_crit = scipy_stats.t.ppf(0.975, len(deltas) - 1)
            ci_low = mean - t_crit * sem
            ci_high = mean + t_crit * sem
            print(f"  95% CI of mean: [{ci_low:+.2f}, {ci_high:+.2f}]")
    
    # Combined T&T
    print(f"\n{'='*100}")
    print("Combined T&T Descriptive Result")
    print(f"{'='*100}")
    
    if paired_results:
        all_deltas = [p["delta_psnr"] for p in paired_results]
        n_total = len(all_deltas)
        n_positive_total = sum(1 for d in all_deltas if d > 0)
        
        print(f"\n  Total paired observations: {n_total}")
        print(f"  Combined mean ΔPSNR: {np.mean(all_deltas):+.2f} dB")
        print(f"  Positive observations: {n_positive_total}/{n_total}")
        
        # Per-scene breakdown
        for scene in SCENES:
            scene_deltas = [p["delta_psnr"] for p in paired_results if p["scene"] == scene]
            scene_pos = sum(1 for d in scene_deltas if d > 0)
            print(f"  {scene}: {scene_pos}/{len(scene_deltas)} positive, mean={np.mean(scene_deltas):+.2f}")
    
    # Save JSON
    output = {
        "runs": all_results,
        "paired": paired_results,
    }
    
    output_path = Path("/mnt/storage_pool/liaoyuanjun/r5_a/r5-a-results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # Gate evaluation
    print(f"\n{'='*100}")
    print("R5-A Gate Evaluation")
    print(f"{'='*100}")
    
    if len(paired_results) == 6:
        train_deltas = [p["delta_psnr"] for p in paired_results if p["scene"] == "train"]
        truck_deltas = [p["delta_psnr"] for p in paired_results if p["scene"] == "truck"]
        
        train_mean = np.mean(train_deltas)
        truck_mean = np.mean(truck_deltas)
        combined_mean = np.mean(all_deltas)
        train_pos = sum(1 for d in train_deltas if d > 0)
        truck_pos = sum(1 for d in truck_deltas if d > 0)
        total_pos = n_positive_total
        
        print(f"\n  train: mean={train_mean:+.2f}, positive={train_pos}/3")
        print(f"  truck: mean={truck_mean:+.2f}, positive={truck_pos}/3")
        print(f"  combined: mean={combined_mean:+.2f}, positive={total_pos}/6")
        
        # PASS_STRONG: both scenes mean > 0 AND ≥ 2/3 positive AND ≥ 5/6 total positive
        if train_mean > 0 and truck_mean > 0 and train_pos >= 2 and truck_pos >= 2 and total_pos >= 5:
            print("\n  *** R5-A = PASS_STRONG ***")
            print("  T&T positive anomaly shows reproducible evidence worthy of causal investigation.")
        # PASS_WEAK: combined mean > 0 but not all conditions for STRONG
        elif combined_mean > 0 and total_pos >= 4:
            print("\n  *** R5-A = PASS_WEAK ***")
            print("  Some reproducibility but not strong. Increase seeds before mechanism development.")
        # FAIL
        else:
            print("\n  *** R5-A = FAIL ***")
            print("  T&T regularization hypothesis unsupported by current evidence.")
            print("  Close T&T regularization branch. Do not execute R5-B.")
    else:
        print(f"\n  INCOMPLETE: {len(paired_results)}/6 paired results available")


if __name__ == "__main__":
    main()
