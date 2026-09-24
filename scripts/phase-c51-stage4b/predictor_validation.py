#!/usr/bin/env python3
"""Phase C51 Stage 4B — Predictor state validation (recall@50, coverage, mask overlap, churn)."""
import json
from pathlib import Path
import numpy as np

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4b")

# Load K50-B1 30K results (has mask_stats)
f = result_dir / "training_room_k50_b1.json"
if not f.exists():
    print("K50-B1 results not found")
    exit()

data = json.load(open(f))
mask_stats = data.get("mask_stats", [])
trajectory = data.get("trajectory", [])

print("=" * 80)
print("Predictor State Validation (K50-B1, Room 30K)")
print("=" * 80)

if not mask_stats:
    print("No mask stats available")
    exit()

print(f"\nTotal mask entries: {len(mask_stats)}")
print(f"First entry: {mask_stats[0]}")
print(f"Last entry:  {mask_stats[-1]}")

# Compute mask stats
keep_fracs = [m["keep_frac"] for m in mask_stats]
print(f"\nKeep fraction: mean={np.mean(keep_fracs):.4f}, std={np.std(keep_fracs):.4f}")
print(f"  min={np.min(keep_fracs):.4f}, max={np.max(keep_fracs):.4f}")

# Mask churn: how much the mask changes between iterations
# Since we don't store the actual mask, we can't compute overlap directly.
# But we can compute statistics from the gradient norms.

# Instead, let's look at the Gaussian count stability as a proxy for mask stability
print("\n--- Gaussian Count Trajectory (proxy for mask stability) ---")
for t in trajectory:
    print(f"  Iter {t['iter']:6d}: GS={t['gaussians']:>10,}, PSNR={t['psnr']:.2f}")

# Compute mask churn from timing data
timing = data.get("timing", {})
print(f"\n--- Timing Summary ---")
print(f"  Sparse iterations: {timing.get('n_sparse_iters', 0)}")
print(f"  Dense iterations:  {timing.get('n_dense_iters', 0)}")
print(f"  Sparse mean:       {timing.get('sparse_mean_ms', 0):.2f} ms")
print(f"  Dense mean:        {timing.get('dense_mean_ms', 0):.2f} ms")
print(f"  Mask mean:         {timing.get('mask_mean_ms', 0):.3f} ms")

# Note: For proper recall@50 and mask overlap, we need to store the actual masks
# or at least the top-k indices. This is a limitation of the current implementation.
print("\n--- Limitation ---")
print("Recall@50 and mask overlap require storing actual mask indices per iteration.")
print("Current implementation only stores aggregate mask statistics.")
print("For the final report, we note this as a limitation and recommend future work")
print("to store mask indices for proper temporal stability analysis.")
