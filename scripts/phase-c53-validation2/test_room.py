#!/usr/bin/env python3
"""Quick test: verify room workload data has current_tiles_mean signal."""
import numpy as np
from pathlib import Path

fp = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-validation2/room_workload.npz")
data = np.load(fp, allow_pickle=True)
keys = sorted(data.keys())

# Check for new signals
print("New workload signals:")
for k in keys:
    if "current_tiles" in k:
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}, mean={data[k].astype(float).mean():.2f}")

print("\nSignal keys at cp500:")
for k in keys:
    if "cp500_s_" in k:
        v = data[k]
        print(f"  {k}: mean={v.astype(float).mean():.4f}, std={v.astype(float).std():.4f}")

print("\nFuture targets at cp500_d50:")
for k in keys:
    if "cp500_d50_" in k:
        v = data[k]
        print(f"  {k}: mean={v.astype(float).mean():.4f}, std={v.astype(float).std():.4f}")

# Quick persistence test
w_cur = data["cp500_s_current_tiles_mean"].astype(np.float64)
w_fut = data["cp500_d50_tiles_mean"].astype(np.float64)
alive = data["cp500_d50_alive"]
mask = alive & (w_cur > 0)
print(f"\nQuick persistence test (cp500, d50):")
print(f"  n_visible={mask.sum()}, n_total={len(w_cur)}")
print(f"  W_current mean={w_cur[mask].mean():.2f}, std={w_cur[mask].std():.2f}")
print(f"  W_future mean={w_fut[mask].mean():.2f}, std={w_fut[mask].std():.2f}")
if np.std(w_cur[mask]) > 1e-12 and np.std(w_fut[mask]) > 1e-12:
    p = float(np.corrcoef(w_cur[mask], w_fut[mask])[0, 1])
    print(f"  Pearson(W_current, W_future) = {p:.3f}")

    # Screen radius comparison
    r_cur = data["cp500_s_screen_radius_mean"].astype(np.float64)
    p_r = float(np.corrcoef(r_cur[mask], w_fut[mask])[0, 1])
    print(f"  Pearson(R_current, W_future) = {p_r:.3f}")

    # OLS R²
    from numpy.linalg import lstsq
    wc = w_cur[mask]
    wf = w_fut[mask]
    rc = r_cur[mask]
    X1 = np.column_stack([np.ones(len(wc)), wc])
    beta1, _, _, _ = lstsq(X1, wf, rcond=None)
    r2_m1 = 1 - np.sum((wf - X1 @ beta1)**2) / np.sum((wf - wf.mean())**2)

    X3 = np.column_stack([np.ones(len(wc)), wc, rc])
    beta3, _, _, _ = lstsq(X3, wf, rcond=None)
    r2_m3 = 1 - np.sum((wf - X3 @ beta3)**2) / np.sum((wf - wf.mean())**2)

    print(f"  R²(M1: W_only) = {r2_m1:.4f}")
    print(f"  R²(M3: W+R)    = {r2_m3:.4f}")
    print(f"  ΔR² = {r2_m3 - r2_m1:.4f}")
