#!/usr/bin/env python3
"""
VOS1 Simple Sequential Runner
Runs training directly without subprocess overhead.
"""

import json
import sys
import time
from pathlib import Path

# Import the training function
sys.path.insert(0, "/root/3dgs-renderer-benchmark")
from vos1_train_minimal import train_visibility_aware

SCENES = ["room", "garden", "bicycle"]
K_VALUES = [1, 2, 4, 8]
ITERATIONS = 30000
SEED = 42

def main():
    output_dir = Path("vos1_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"VOS1 Benchmark: {len(SCENES)} scenes × {len(K_VALUES)} k values")
    print(f"Iterations: {ITERATIONS}, Seed: {SEED}")
    print(f"Output: {output_dir}")
    print()
    
    all_results = {}
    
    for scene in SCENES:
        all_results[scene] = {}
        
        for k in K_VALUES:
            print(f"\n{'='*60}")
            print(f"Training {scene} k={k}")
            print(f"{'='*60}")
            
            scene_path = f"data/official/mipnerf360/{scene}"
            
            start_time = time.time()
            
            try:
                model, metrics = train_visibility_aware(
                    scene_path, k, ITERATIONS, SEED, str(output_dir)
                )
                
                elapsed = time.time() - start_time
                
                result = {
                    "scene": scene,
                    "k": k,
                    "iterations": ITERATIONS,
                    "seed": SEED,
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "optimizer_time": metrics["optimizer_time"],
                        "total_updates": metrics["total_updates"],
                        "total_skipped": metrics["total_skipped"],
                        "skip_rate": metrics["total_skipped"] / (metrics["total_updates"] + metrics["total_skipped"]) if (metrics["total_updates"] + metrics["total_skipped"]) > 0 else 0,
                    },
                    "wall_time_actual": elapsed,
                }
                
                all_results[scene][k] = result
                
                # Save individual result
                result_file = output_dir / f"vos1_{scene}_k{k}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                print(f"\nCompleted {scene} k={k} in {elapsed:.1f}s")
                print(f"  Wall time: {metrics['wall_time']:.1f}s")
                print(f"  Optimizer time: {metrics['optimizer_time']:.1f}s")
                print(f"  Total updates: {metrics['total_updates']:,}")
                print(f"  Total skipped: {metrics['total_skipped']:,}")
                print(f"  Skip rate: {100*result['metrics']['skip_rate']:.1f}%")
                
            except Exception as e:
                print(f"ERROR training {scene} k={k}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save all results
    print(f"\n{'='*60}")
    print("Saving final results...")
    print(f"{'='*60}")
    
    final_results = {
        "config": {
            "scenes": SCENES,
            "k_values": K_VALUES,
            "iterations": ITERATIONS,
            "seed": SEED,
        },
        "results": all_results,
    }
    
    with open(output_dir / "vos1_all_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("VOS1 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for scene in SCENES:
        if scene not in all_results or 1 not in all_results[scene]:
            continue
        
        print(f"\n{sene}:")
        baseline = all_results[scene][1]["metrics"]
        print(f"  Baseline (k=1): {baseline['wall_time']:.1f}s wall, "
              f"{baseline['optimizer_time']:.1f}s opt, "
              f"{baseline['total_updates']:,} updates")
        
        for k in [2, 4, 8]:
            if k not in all_results[scene]:
                continue
            
            m = all_results[scene][k]["metrics"]
            speedup = baseline["wall_time"] / m["wall_time"]
            opt_reduction = 1.0 - (m["optimizer_time"] / baseline["optimizer_time"])
            
            print(f"  k={k}: {m['wall_time']:.1f}s wall ({speedup:.2f}×), "
                  f"{opt_reduction*100:.1f}% opt reduction, "
                  f"{m['skip_rate']*100:.1f}% skip rate")
    
    print(f"\nResults saved to {output_dir / 'vos1_all_results.json'}")

if __name__ == "__main__":
    main()
