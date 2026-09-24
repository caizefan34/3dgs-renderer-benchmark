#!/usr/bin/env python3
"""Compare B1/P/A results from saved .pt files + re-run benchmarks."""
import sys, os, json, subprocess
import torch
import numpy as np

RESULTS_DIR = "/tmp/accutile_a100_results"

def tensor_metrics(a, b):
    a, b = a.detach(), b.detach()
    diff = (a - b).abs()
    norm_b = b.flatten().norm().item()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "relative_L2": (diff.flatten().norm().item() / norm_b) if norm_b > 0 else float('inf'),
    }

def main():
    # Load saved tensors
    b1 = torch.load(f"{RESULTS_DIR}/variant_B1.pt", map_location="cpu")
    p = torch.load(f"{RESULTS_DIR}/variant_P.pt", map_location="cpu")
    a = torch.load(f"{RESULTS_DIR}/variant_A.pt", map_location="cpu")

    print("=" * 70)
    print("TENSOR-LEVEL CORRECTNESS (B1/P/A)")
    print("=" * 70)

    results = {}
    for name, v in [("P", p), ("A", a)]:
        print(f"\n{name} vs B1:")
        rgb_m = tensor_metrics(b1["rgb"], v["rgb"])
        alpha_m = tensor_metrics(b1["alpha"], v["alpha"])
        print(f"  RGB: max_abs={rgb_m['max_abs']:.4e}, mean_abs={rgb_m['mean_abs']:.4e}, rel_L2={rgb_m['relative_L2']:.4e}")
        print(f"  alpha: max_abs={alpha_m['max_abs']:.4e}, mean_abs={alpha_m['mean_abs']:.4e}, rel_L2={alpha_m['relative_L2']:.4e}")

        grad_results = {}
        for gname in ("means", "quats", "scales", "opacities", "colors"):
            if gname in b1["grads"] and gname in v["grads"]:
                gm = tensor_metrics(b1["grads"][gname], v["grads"][gname])
                print(f"  grad {gname}: max_abs={gm['max_abs']:.4e}, mean_abs={gm['mean_abs']:.4e}, rel_L2={gm['relative_L2']:.4e}")
                grad_results[gname] = gm
        results[name] = {
            "rgb": rgb_m, "alpha": alpha_m, "grads": grad_results,
            "intersections": v["intersections"],
            "rgb_sum": v["rgb_sum"], "alpha_sum": v["alpha_sum"],
        }

    print(f"\n{'='*70}")
    print("INTERSECTION COUNTS")
    print(f"{'='*70}")
    print(f"B1: {b1['intersections']:,}")
    print(f"P:  {p['intersections']:,}  (reduction={1-p['intersections']/b1['intersections']:.4f})")
    print(f"A:  {a['intersections']:,}  (reduction={1-a['intersections']/b1['intersections']:.4f})")

    # Save correctness results
    output = {
        "scene": "room",
        "B1": {"intersections": b1["intersections"], "rgb_sum": b1["rgb_sum"], "alpha_sum": b1["alpha_sum"]},
        "P": results["P"],
        "A": results["A"],
        "benchmark": {},  # Will be filled by benchmark script
    }

    with open(f"{RESULTS_DIR}/accutile_b1pa_correctness.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/accutile_b1pa_correctness.json")

if __name__ == "__main__":
    main()
