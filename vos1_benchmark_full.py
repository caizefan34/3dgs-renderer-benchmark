#!/usr/bin/env python3
"""
VOS1 Full Benchmark Runner
Runs baseline (k=1) and visibility-aware (k=2,4,8) across all scenes.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

SCENES = ["room", "garden", "bicycle"]
K_VALUES = [1, 2, 4, 8]
ITERATIONS = 30000
SEED = 42

def run_training(scene, k, output_dir, gpu=0):
    """Run training for a single scene and k value."""
    scene_path = f"data/official/mipnerf360/{scene}"
    
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    
    if k == 1:
        # Baseline: use k=1 (all visible, no skipping)
        cmd = [
            "python", "vos1_train_minimal.py",
            "--scene", scene_path,
            "--k", "1",
            "--iterations", str(ITERATIONS),
            "--seed", str(SEED),
            "--output", output_dir,
        ]
    else:
        cmd = [
            "python", "vos1_train_minimal.py",
            "--scene", scene_path,
            "--k", str(k),
            "--iterations", str(ITERATIONS),
            "--seed", str(SEED),
            "--output", output_dir,
        ]
    
    print(f"Running {scene} k={k} on GPU {gpu}...")
    start = time.time()
    
    result = subprocess.run(
        cmd,
        cwd="/root/3dgs-renderer-benchmark",
        capture_output=True,
        text=True,
        env={**env, "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"ERROR: {scene} k={k} failed")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return None
    
    # Load results
    if k == 1:
        result_file = Path(output_dir) / f"vos1_{scene}_k1.json"
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
    """Compute speedup, workload reduction, and quality deltas."""
    metrics = {}
    
    for scene in SCENES:
        if scene not in all_results or 1 not in all_results[scene]:
            continue
            
        baseline = all_results[scene][1]
        baseline_metrics = baseline["metrics"]
        
        metrics[scene] = {
            "baseline": {
                "wall_time": baseline_metrics["wall_time"],
                "optimizer_time": baseline_metrics["optimizer_time"],
                "total_updates": baseline_metrics["total_updates"],
                "total_skipped": baseline_metrics["total_skipped"],
            }
        }
        
        for k in [2, 4, 8]:
            if k not in all_results[scene]:
                continue
            
            va = all_results[scene][k]
            va_metrics = va["metrics"]
            
            # Speedup
            speedup = baseline_metrics["wall_time"] / va_metrics["wall_time"]
            
            # Optimizer workload reduction
            opt_reduction = 1.0 - (va_metrics["optimizer_time"] / baseline_metrics["optimizer_time"])
            
            # Update reduction
            total_possible = baseline_metrics["total_updates"]
            actual_updates = va_metrics["total_updates"]
            update_reduction = 1.0 - (actual_updates / total_possible) if total_possible > 0 else 0
            
            # Skip rate
            skip_rate = va_metrics.get("skip_rate", 0)
            
            metrics[scene][f"k{k}"] = {
                "wall_time": va_metrics["wall_time"],
                "optimizer_time": va_metrics["optimizer_time"],
                "total_updates": actual_updates,
                "total_skipped": va_metrics.get("total_skipped", 0),
                "skip_rate": skip_rate,
                "speedup": speedup,
                "optimizer_workload_reduction": opt_reduction,
                "update_reduction": update_reduction,
            }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="vos1_benchmark_results")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"VOS1 Benchmark: {len(SCENES)} scenes × {len(K_VALUES)} k values")
    print(f"Iterations: {args.iterations}, Seed: {SEED}")
    print(f"Output: {output_dir}")
    print()
    
    # GPU assignment for parallel execution
    gpu_assignments = [
        ("room", 1, 0),
        ("room", 2, 4),
        ("room", 4, 5),
        ("room", 8, 6),
        ("garden", 1, 0),
        ("garden", 2, 4),
        ("garden", 4, 5),
        ("garden", 8, 6),
        ("bicycle", 1, 0),
        ("bicycle", 2, 4),
        ("bicycle", 4, 5),
        ("bicycle", 8, 6),
    ]
    
    # Run all configurations
    all_results = {scene: {} for scene in SCENES}
    
    for scene, k, gpu in gpu_assignments:
        result = run_training(scene, k, str(output_dir), gpu)
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
                "scenes": SCENES,
                "k_values": K_VALUES,
                "iterations": args.iterations,
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
    
    for scene in SCENES:
        if scene not in metrics:
            continue
        print(f"\n{sene}:")
        baseline = metrics[scene]["baseline"]
        print(f"  Baseline (k=1): {baseline['wall_time']:.1f}s wall, "
              f"{baseline['optimizer_time']:.1f}s opt, "
              f"{baseline['total_updates']:,} updates")
        
        for k in [2, 4, 8]:
            if f"k{k}" not in metrics[scene]:
                continue
            
            m = metrics[scene][f"k{k}"]
            print(f"  k={k}: {m['wall_time']:.1f}s wall ({m['speedup']:.2f}×), "
                  f"{m['optimizer_workload_reduction']*100:.1f}% opt reduction, "
                  f"{m['skip_rate']*100:.1f}% skip rate")
    
    return metrics


if __name__ == "__main__":
    main()
