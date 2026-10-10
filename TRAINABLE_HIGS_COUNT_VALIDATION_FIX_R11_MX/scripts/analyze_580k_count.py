#!/usr/bin/env python3
"""R11: Analyze 580K intersection count"""
import numpy as np

print("=== R11: Intersection Count Analysis ===\n")

# 580K measurements
n_gaussians = 580_000
n_visible = 273_234
n_isects = 490_387_290

print(f"At 580K Gaussians (B1):")
print(f"  Total Gaussians: {n_gaussians:,}")
print(f"  Visible after culling: {n_visible:,}")
print(f"  Actual n_isects: {n_isects:,}")
print()

# Calculate per-Gaussian coverage
avg_isects_per_visible = n_isects / n_visible
avg_isects_per_total = n_isects / n_gaussians

print(f"Intersection coverage:")
print(f"  Per visible Gaussian: {avg_isects_per_visible:.2f}")
print(f"  Per total Gaussian: {avg_isects_per_total:.2f}")
print()

# Expected for 540p (33x17 tiles = 561 tiles)
# If each Gaussian covers ~9 tiles on average
expected_isects = n_visible * 9
print(f"Expected (9 tiles/Gaussian): {expected_isects:,}")
print(f"Actual: {n_isects:,}")
print(f"Ratio: {n_isects / expected_isects:.2f}x")
print()

# If each visible Gaussian covers all 561 tiles
max_isects = n_visible * 561
print(f"Max possible (561 tiles/Gaussian): {max_isects:,}")
print(f"Actual / Max: {n_isects / max_isects * 100:.2f}%")
print()

# Extrapolate to 4.4M
print(f"Extrapolation to 4.4M:")
scale_factor = 4_413_928 / 580_000
n_isects_4p4m_est = int(n_isects * scale_factor)
print(f"  Scale factor: {scale_factor:.2f}x")
print(f"  Estimated n_isects at 4.4M: {n_isects_4p4m_est:,}")
print()

# Compare with R10 OOM
oom_n_isects = 9_957_363_285
print(f"Compare with R10 OOM:")
print(f"  R10 n_isects: {oom_n_isects:,}")
print(f"  Estimated: {n_isects_4p4m_est:,}")
print(f"  R10 / Estimated: {oom_n_isects / n_isects_4p4m_est:.2f}x")
