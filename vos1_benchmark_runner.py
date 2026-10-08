#!/usr/bin/env python3
"""
VOS1 Benchmark Runner
Executes baseline (k=1) and visibility-aware (k=2,4,8) training across scenes.
Computes speedup, workload reduction, and quality deltas.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

SCENES = ["room", "garden", "bicycle"]
K_VALUES = [1, 2, 4, 8]  # 1=baseline, 2/4/8=visibility-aware
ITERATIONS = 30000
SEED = 42

def run_training(scene, k, output_dir, epic05_host="EPIC-05"):
    """
    Run training for a single scene and k value.
    
    For k=1: use baseline script
    For k=2,4,8: use visibility-aware script
    """
    if k == 1:
        script = "vos1_baseline_train.py"
        cmd = [
            "ssh", epic05_host,
            f"cd /root/3dgs-renderer-benchmark &&",
            f"python {script}",
            f"--scene data/official/mipnerf360/{scene}",
            f"--iterations {ITERATIONS}",
            f"--seed {SEED}",
            f"--output {output_dir}"
        ]
    else:
        script = "vos1_visibility_aware_train.py"
        cmd = [
            "ssh", epic05_host,
            f"cd /root/3dgs-renderer-benchmark &&",
            f"python {script}",
            f"--scene data/official/mipnerf360/{scene}",
            f"--k {k}",
            f"--iterations {ITERATIONS}",
            f"--seed {SEED}",
            f"--output {output_dir}"
        ]
    
    print(f"Running {scene} k={k}...")
    start = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"ERROR: {scene} k={k} failed")
        print(result.stderr)
        return None
    
    # Load results
    if k == 1:
        result_file = Path(output_dir) / f"vos1_baseline_{scene}.json"
    else:
        result_file = Path(output_dir) / f"vos1_{scene}_k{k}.json"
    
    if not result_file.exists():
        print(f"ERROR: Result file not found: {result_file}")
        return None
    
    with open(result_file) as f:
        results = json.load(f)
    
    results["wall_time_actual"] = elapsed
    return results


def compute_metrics(all_results):
    """
    Compute speedup, workload reduction, and quality deltas.
    
    Pass criteria:
    - PSNR loss <= 0.05dB
    - AND (speedup >= 1.2x OR optimizer workload reduction >= 25%)
    """
    metrics = {}
    
    for scene in SCENES:
        baseline = all_results[scene][1]  # k=1
        baseline_metrics = baseline["metrics"]
        baseline_quality = baseline["quality"]
        
        metrics[scene] = {
            "baseline": {
                "wall_time": baseline_metrics["wall_time"],
                "optimizer_time": baseline_metrics["optimizer_time"],
                "total_updates": baseline_metrics["total_updates"],
                "psnr": baseline_quality["psnr"],
                "ssim": baseline_quality["ssim"],
                "lpips": baseline_quality["lpips"],
            }
        }
        
        for k in [2, 4, 8]:
            if k not in all_results[scene]:
                continue
            
            va = all_results[scene][k]
            va_metrics = va["metrics"]
            va_quality = va["quality"]
            
            # Speedup
            speedup = baseline_metrics["wall_time"] / va_metrics["wall_time"]
            
            # Optimizer workload reduction
            opt_reduction = 1.0 - (va_metrics["optimizer_time"] / baseline_metrics["optimizer_time"])
            
            # Update reduction (skipped updates / total possible updates)
            total_possible = baseline_metrics["total_updates"]
            actual_updates = va_metrics["total_updates"]
            update_reduction = 1.0 - (actual_updates / total_possible)
            
            # Quality deltas
            psnr_loss = baseline_quality["psnr"] - va_quality["psnr"]
            ssim_loss = baseline_quality["ssim"] - va_quality["ssim"]
            lpips_increase = va_quality["lpips"] - baseline_quality["lpips"]
            
            # Pass criteria
            psnr_pass = psnr_loss <= 0.05
            speedup_pass = speedup >= 1.2
            workload_pass = opt_reduction >= 0.25 or update_reduction >= 0.25
            overall_pass = psnr_pass and (speedup_pass or workload_pass)
            
            metrics[scene][f"k{k}"] = {
                "wall_time": va_metrics["wall_time"],
                "optimizer_time": va_metrics["optimizer_time"],
                "total_updates": actual_updates,
                "total_skipped": va_metrics.get("total_skipped", 0),
                "speedup": speedup,
                "optimizer_workload_reduction": opt_reduction,
                "update_reduction": update_reduction,
                "psnr": va_quality["psnr"],
                "ssim": va_quality["ssim"],
                "lpips": va_quality["lpips"],
                "psnr_loss": psnr_loss,
                "ssim_loss": ssim_loss,
                "lpips_increase": lpips_increase,
                "psnr_pass": psnr_pass,
                "speedup_pass": speedup_pass,
                "workload_pass": workload_pass,
                "overall_pass": overall_pass,
            }
    
    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, default="vos1_results")
    parser.add_argument("--scenes", type=str, nargs="+", default=SCENES)
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"VOS1 Benchmark: {len(args.scenes)} scenes × {len(args.k_values)} k values")
    print(f"Output directory: {output_dir}")
    print()
    
    # Run all configurations
    all_results = {scene: {} for scene in args.scenes}
    
    for scene in args.scenes:
        for k in args.k_values:
            result = run_training(scene, k, str(output_dir))
            if result:
                all_results[scene][k] = result
    
    # Compute metrics
    print()
    print("Computing metrics...")
    metrics = compute_metrics(all_results)
    
    # Save results
    results_file = output_dir / "vos1_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "scenes": args.scenes,
                "k_values": args.k_values,
                "iterations": ITERATIONS,
                "seed": SEED,
            },
            "results": all_results,
            "metrics": metrics,
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Print summary
    print()
    print("=" * 80)
    print("VOS1 BENCHMARK SUMMARY")
    print("=" * 80)
    
    for scene in args.scenes:
        print(f"
{sene}:")
        baseline = metrics[scene]["baseline"]
        print(f"  Baseline: {baseline['wall_time']:.1f}s, PSNR={baseline['psnr']:.2f}dB")
        
        for k in [2, 4, 8]:
            if f"k{k}" not in metrics[scene]:
                continue
            
            m = metrics[scene][f"k{k}"]
            status = "PASS" if m["overall_pass"] else "FAIL"
            print(f"  k={k}: {status} | Speedup={m['speedup']:.2f}× | "
                  f"Opt reduction={100*m['optimizer_workload_reduction']:.1f}% | "
                  f"PSNR loss={m['psnr_loss']:.3f}dB")


if __name__ == "__main__":
    main()
