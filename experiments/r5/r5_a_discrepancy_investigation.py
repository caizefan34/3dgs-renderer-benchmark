#!/usr/bin/env python3
"""Investigate R4 vs R5 number discrepancy."""
import json, os, sys
from pathlib import Path

R4_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")
R5_BASE = Path("/mnt/storage_pool/liaoyuanjun/r5_a")

print("=" * 100)
print("PART 1: R4 final_results.json")
print("=" * 100)

fr_path = R4_BASE / "final_results.json"
if fr_path.exists():
    d = json.load(open(fr_path))
    print(f"Type: {type(d)}")
    if isinstance(d, dict):
        print(f"Keys: {list(d.keys())}")
        for key in ["train", "truck"]:
            if key in d:
                print(f"\n--- {key} ---")
                print(json.dumps(d[key], indent=2)[:1000])
    elif isinstance(d, list):
        print(f"Length: {len(d)}")
        for item in d:
            if isinstance(item, dict) and item.get("scene") in ["train", "truck"]:
                print(f"\n--- {item.get('scene')} ---")
                print(json.dumps(item, indent=2)[:1000])
else:
    print("final_results.json NOT FOUND")

print("\n" + "=" * 100)
print("PART 2: R4 training_metrics.json for train/truck")
print("=" * 100)

for scene in ["train", "truck"]:
    for method in ["baseline", "candidate_c"]:
        run_dir = R4_BASE / scene / method
        tm_path = run_dir / "training_metrics.json"
        qb_path = run_dir / "quality_baseline.json"
        
        print(f"\n--- {scene}/{method} ---")
        
        if tm_path.exists():
            tm = json.load(open(tm_path))
            ckpts = tm.get("checkpoints", {})
            print(f"  training_metrics.json checkpoints: {sorted(ckpts.keys(), key=int)}")
            # Print last few
            for it in sorted(ckpts.keys(), key=int)[-3:]:
                c = ckpts[it]
                print(f"    iter {it}: PSNR={c.get('psnr')}, SSIM={c.get('ssim')}, N={c.get('N_gaussians')}")
        else:
            print(f"  training_metrics.json NOT FOUND")
        
        if qb_path.exists():
            qb = json.load(open(qb_path))
            print(f"  quality_baseline.json: {json.dumps(qb, indent=2)[:500]}")
        else:
            print(f"  quality_baseline.json NOT FOUND")

print("\n" + "=" * 100)
print("PART 3: R5 aggregation extraction (what r5_a_aggregate.py read)")
print("=" * 100)

for scene in ["train", "truck"]:
    for method in ["baseline", "candidate_c"]:
        run_dir = R4_BASE / scene / method
        tm_path = run_dir / "training_metrics.json"
        
        print(f"\n--- {scene}/{method} (R4 dir, seed=0) ---")
        if tm_path.exists():
            tm = json.load(open(tm_path))
            ckpts = tm.get("checkpoints", {})
            # The aggregation script reads ckpts.get("30000")
            c30k = ckpts.get("30000", {})
            print(f"  30K checkpoint: PSNR={c30k.get('psnr')}, SSIM={c30k.get('ssim')}, N={c30k.get('N_gaussians')}")
        else:
            print(f"  training_metrics.json NOT FOUND")

print("\n" + "=" * 100)
print("PART 4: Check for R4 aggregate_results.py output / log files")
print("=" * 100)

# Check the R4 aggregate script to see what it reads
agg_path = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/experiments/r4/aggregate_results.py")
if agg_path.exists():
    print(f"  aggregate_results.py found at {agg_path}")
    with open(agg_path) as f:
        content = f.read()
    # Look for what files it reads
    for keyword in ["quality_baseline", "training_metrics", "log", "json", "PSNR", "N_gaussians", "final"]:
        lines = [l.strip() for l in content.split('\n') if keyword.lower() in l.lower()]
        if lines:
            print(f"\n  Lines mentioning '{keyword}':")
            for l in lines[:5]:
                print(f"    {l}")
else:
    print("  aggregate_results.py NOT FOUND")

print("\n" + "=" * 100)
print("PART 5: R4 log files — last PSNR eval lines")
print("=" * 100)

for scene in ["train", "truck"]:
    for method in ["baseline", "candidate_c"]:
        # Try multiple log paths
        log_paths = [
            f"/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs/{scene}_{method}.log",
            f"/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/{scene}/{method}/training.log",
        ]
        
        print(f"\n--- {scene}/{method} ---")
        for lp in log_paths:
            if os.path.exists(lp):
                print(f"  Log: {lp}")
                with open(lp) as f:
                    lines = f.readlines()
                # Find all PSNR lines
                psnr_lines = [l.strip() for l in lines if "PSNR=" in l]
                print(f"  Total PSNR eval lines: {len(psnr_lines)}")
                for l in psnr_lines[-3:]:
                    print(f"    {l}")
                break
        else:
            print(f"  No log file found")
