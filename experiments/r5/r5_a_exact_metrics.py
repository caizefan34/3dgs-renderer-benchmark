#!/usr/bin/env python3
"""Get exact full-precision metrics for all 12 runs."""
import json, os
from pathlib import Path

R4_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")
R5_BASE = Path("/mnt/storage_pool/liaoyuanjun/r5_a")

print("=" * 100)
print("EXACT METRICS — All 12 runs")
print("=" * 100)

# Seed 0 from final_results.json (R4 original)
print("\n--- SEED 0: final_results.json (R4 original, mtime 2026-09-18 23:58) ---")
fr = json.load(open(R4_BASE / "final_results.json"))
for item in fr:
    if item["scene"] in ["train", "truck"]:
        print(f"  {item['scene']} B: PSNR={item['baseline_psnr']:.10f}, SSIM={item['baseline_ssim']:.10f}, N={item['baseline_n']}")
        print(f"  {item['scene']} C: PSNR={item['candidate_psnr']:.10f}, SSIM={item['candidate_ssim']:.10f}, N={item['candidate_n']}")

# Seed 0 from training_metrics.json (re-run, mtime 2026-09-19 01:04)
print("\n--- SEED 0: training_metrics.json (re-run, mtime 2026-09-19 01:04) ---")
for scene in ["train", "truck"]:
    for method in ["baseline", "candidate_c"]:
        tm = json.load(open(R4_BASE / scene / method / "training_metrics.json"))
        c30k = tm["checkpoints"]["30000"]
        print(f"  {scene} {method}: PSNR={c30k['psnr']:.10f}, SSIM={c30k['ssim']:.10f}, N={c30k['N_gaussians']}")

# Seeds 1, 2 from R5-A training_metrics.json
print("\n--- SEEDS 1, 2: R5-A training_metrics.json ---")
for seed in [1, 2]:
    for scene in ["train", "truck"]:
        for method in ["baseline", "candidate_c"]:
            run_dir = R5_BASE / f"{scene}_seed{seed}" / method
            tm_path = run_dir / "training_metrics.json"
            if tm_path.exists():
                tm = json.load(open(tm_path))
                c30k = tm["checkpoints"].get("30000", {})
                if c30k:
                    print(f"  {scene} seed={seed} {method}: PSNR={c30k['psnr']:.10f}, SSIM={c30k['ssim']:.10f}, N={c30k['N_gaussians']}")
                else:
                    max_iter = max(tm["checkpoints"].keys(), key=int)
                    c = tm["checkpoints"][max_iter]
                    print(f"  {scene} seed={seed} {method}: PSNR={c['psnr']:.10f} (iter {max_iter}, NOT 30K!)")
            else:
                print(f"  {scene} seed={seed} {method}: training_metrics.json NOT FOUND")

# Compute deltas both ways
print("\n" + "=" * 100)
print("PAIRED DELTAS — Using final_results.json for seed 0 (R4 original)")
print("=" * 100)

seed0_orig = {}
for item in fr:
    if item["scene"] in ["train", "truck"]:
        seed0_orig[item["scene"]] = item

for scene in ["train", "truck"]:
    deltas = []
    # Seed 0 (original)
    d0 = seed0_orig[scene]["candidate_psnr"] - seed0_orig[scene]["baseline_psnr"]
    deltas.append(d0)
    print(f"  {scene} seed=0 (orig): B={seed0_orig[scene]['baseline_psnr']:.4f} C={seed0_orig[scene]['candidate_psnr']:.4f} Δ={d0:+.4f}")
    
    # Seeds 1, 2
    for seed in [1, 2]:
        b_tm = json.load(open(R5_BASE / f"{scene}_seed{seed}" / "baseline" / "training_metrics.json"))
        c_tm = json.load(open(R5_BASE / f"{scene}_seed{seed}" / "candidate_c" / "training_metrics.json"))
        b = b_tm["checkpoints"]["30000"]
        c = c_tm["checkpoints"]["30000"]
        d = c["psnr"] - b["psnr"]
        deltas.append(d)
        print(f"  {scene} seed={seed}: B={b['psnr']:.4f} C={c['psnr']:.4f} Δ={d:+.4f}")
    
    import numpy as np
    print(f"  → Mean={np.mean(deltas):+.4f}, Positive={sum(1 for d in deltas if d > 0)}/3")

print("\n" + "=" * 100)
print("PAIRED DELTAS — Using training_metrics.json for seed 0 (re-run)")
print("=" * 100)

for scene in ["train", "truck"]:
    deltas = []
    # Seed 0 (re-run)
    b_tm = json.load(open(R4_BASE / scene / "baseline" / "training_metrics.json"))
    c_tm = json.load(open(R4_BASE / scene / "candidate_c" / "training_metrics.json"))
    b = b_tm["checkpoints"]["30000"]
    c = c_tm["checkpoints"]["30000"]
    d0 = c["psnr"] - b["psnr"]
    deltas.append(d0)
    print(f"  {scene} seed=0 (rerun): B={b['psnr']:.4f} C={c['psnr']:.4f} Δ={d0:+.4f}")
    
    # Seeds 1, 2
    for seed in [1, 2]:
        b_tm = json.load(open(R5_BASE / f"{scene}_seed{seed}" / "baseline" / "training_metrics.json"))
        c_tm = json.load(open(R5_BASE / f"{scene}_seed{seed}" / "candidate_c" / "training_metrics.json"))
        b = b_tm["checkpoints"]["30000"]
        c = c_tm["checkpoints"]["30000"]
        d = c["psnr"] - b["psnr"]
        deltas.append(d)
        print(f"  {scene} seed={seed}: B={b['psnr']:.4f} C={c['psnr']:.4f} Δ={d:+.4f}")
    
    print(f"  → Mean={np.mean(deltas):+.4f}, Positive={sum(1 for d in deltas if d > 0)}/3")

# SSIM deltas both ways
print("\n" + "=" * 100)
print("SSIM DELTAS — Both versions")
print("=" * 100)

for label, use_orig in [("final_results (orig)", True), ("training_metrics (rerun)", False)]:
    print(f"\n--- {label} ---")
    all_ssim_deltas = []
    for scene in ["train", "truck"]:
        for seed in [0, 1, 2]:
            if seed == 0:
                if use_orig:
                    b_ssim = seed0_orig[scene]["baseline_ssim"]
                    c_ssim = seed0_orig[scene]["candidate_ssim"]
                else:
                    b_tm = json.load(open(R4_BASE / scene / "baseline" / "training_metrics.json"))
                    c_tm = json.load(open(R4_BASE / scene / "candidate_c" / "training_metrics.json"))
                    b_ssim = b_tm["checkpoints"]["30000"]["ssim"]
                    c_ssim = c_tm["checkpoints"]["30000"]["ssim"]
            else:
                b_tm = json.load(open(R5_BASE / f"{scene}_seed{seed}" / "baseline" / "training_metrics.json"))
                c_tm = json.load(open(R5_BASE / f"{scene}_seed{seed}" / "candidate_c" / "training_metrics.json"))
                b_ssim = b_tm["checkpoints"]["30000"]["ssim"]
                c_ssim = c_tm["checkpoints"]["30000"]["ssim"]
            
            d_ssim = c_ssim - b_ssim
            all_ssim_deltas.append(d_ssim)
            print(f"  {scene} seed={seed}: ΔSSIM={d_ssim:+.6f} {'(neg)' if d_ssim < 0 else '(pos)'}")
    
    n_neg = sum(1 for d in all_ssim_deltas if d < 0)
    n_pos = sum(1 for d in all_ssim_deltas if d > 0)
    print(f"  → Negative: {n_neg}/6, Positive: {n_pos}/6")
    
    # Exact binomial test
    from math import comb
    # One-sided: P(all 6 negative | p=0.5) = (0.5)^6 = 1/64
    p_one_sided = (0.5)**6
    # Two-sided: 2 * P(all 6 negative) = 2/64 = 1/32
    p_two_sided = 2 * p_one_sided
    print(f"  → One-sided exact p (all negative): {p_one_sided:.6f} = 1/64")
    print(f"  → Two-sided exact p: {p_two_sided:.6f} = 1/32 = 0.03125")
