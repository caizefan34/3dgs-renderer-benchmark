#!/usr/bin/env python3
"""
C35-1: Adaptive Hybrid Blend Backend — Tile Density Analysis.

Uses existing C33-D workload data to analyze tile-level density distribution.
No GPU run needed — purely analytical from observation data.
"""
import json, sys, math
import numpy as np

path = "results/phase-c31/c33_d_workload_data.json"
with open(path) as f:
    data = json.load(f)
recs = data["records"][1:]  # skip iter 0 (cold start outlier)

# ─── HiGS macro-tile geometry (tile_size=16, 1080p) ───
tile_size = 16
width, height = 1920, 1080
tile_w = math.ceil(width / tile_size)   # 120
tile_h = math.ceil(height / tile_size)  # 68
n_tiles = tile_w * tile_h  # 8160

# HiGS uses macro-tiles of size 8x8 = 64 fine tiles
MACRO_TILE = 8  # fine tiles per macro-tile dimension
macro_w = math.ceil(tile_w / MACRO_TILE)   # 15
macro_h = math.ceil(tile_h / MACRO_TILE)   # 9
n_macro = macro_w * macro_h  # 135

fine_per_macro = MACRO_TILE * MACRO_TILE  # 64

n_vis = np.array([r["n_visible"] for r in recs], dtype=float)
n_isect = np.array([r["n_intersections"] for r in recs], dtype=float)
n_active_tiles = np.array([r["n_active_tiles"] for r in recs], dtype=float)
fwd_ms = np.array([r["fwd_ms"] for r in recs], dtype=float)
total_render = np.array([r["total_render_ms"] for r in recs], dtype=float)

print("=== C35-1: Tile Density Analysis ===")
print(f"Image: {width}x{height}, tile_size={tile_size}")
print(f"Fine tiles: {tile_w}x{tile_h} = {n_tiles}")
print(f"Macro-tiles: {macro_w}x{macro_h} = {n_macro} (each = {MACRO_TILE}x{MACRO_TILE}=64 fine tiles)")
print(f"Iterations analyzed: {len(recs)}")

# ─── Density distribution across tiles ───
# From C33-D data: n_active_tiles = tiles with >= 1 intersection
# We can estimate: average intersections per active tile = n_isect / n_active_tiles
avg_isect_per_active = n_isect / n_active_tiles
avg_isect_total = n_isect / n_tiles

print(f"\n--- Tile Density Stats ---")
print(f"  Active tiles (>=1 GS):")
print(f"    mean={np.mean(n_active_tiles):.0f} / {n_tiles} = {np.mean(n_active_tiles)/n_tiles*100:.1f}%")
print(f"    min={np.min(n_active_tiles):.0f}  max={np.max(n_active_tiles):.0f}")
print(f"  Inactive tiles (empty): {n_tiles - np.max(n_active_tiles):.0f}-{n_tiles - np.min(n_active_tiles):.0f}")
print(f"  Intersections per active tile:")
print(f"    mean={np.mean(avg_isect_per_active):.1f}  median={np.median(avg_isect_per_active):.1f}")
print(f"    min={np.min(avg_isect_per_active):.1f}  max={np.max(avg_isect_per_active):.1f}")
print(f"  Intersections per total tile:")
print(f"    mean={np.mean(avg_isect_total):.1f}")

# Estimate GS per tile geometry:
# Each visible GS covers ~3 tiles on average (from C33-D radii data: avg radius ~3-5 pixels)
# So active_tiles ~ n_visible * 3 / 8160 roughly?
tile_coverage = n_active_tiles / n_vis
print(f"\n  Active tiles per visible GS:")
print(f"    mean={np.mean(tile_coverage):.4f}  max={np.max(tile_coverage):.4f}  min={np.min(tile_coverage):.4f}")

# ─── Sparse vs Dense tile analysis ───
# A tile is "dense" if it has enough intersections to benefit from GEMM
# GEMM blending: O(Gs_per_tile^2) vs warp blending: O(Gs_per_tile)
# Crossover at roughly 32-64 GS/tile

# From per-tile density, we can estimate the histogram if we assume 
# a roughly exponential distribution (most tiles sparse, few dense)

# Given: n_isect intersections spread across n_active_tiles
# Under uniform distribution: isect_per_tile = n_isect / n_active_tiles
# but real distributions are heavy-tailed

# Let's compute what fraction would be in "dense" tiles if we set
# thresholds at K = 32, 64, 128, 256 GS/tile

for threshold in [32, 64, 128, 256]:
    # If distribution were uniform, isect_per_active ≈ constant
    # but real: most active tiles have 1-5 GS, some have 200+
    # The HiGS macro-tile approach already bins 64 fine tiles together
    
    # Intersections per MACRO-tile (64 fine tiles)
    isect_per_macro_est = n_isect / n_macro
    
    isect_per_macro_active = n_isect / (n_macro * np.ones_like(n_isect))  # assume all macro-tiles active
    
    print(f"\n--- Threshold K={threshold} GS/tile ---")
    # Under uniform: no tile exceeds mean
    # Under realistic: some tiles are dense
    
    # Estimate from intersection count:
    # If we have I intersections and T active tiles,
    # the top 5% of tiles carry ~30-40% of intersections (Pareto-like)
    pareto_top_pct = 0.05
    top_tiles = max(1, int(n_active_tiles * pareto_top_pct))
    # Estimated: top 5% tiles have 35% of GS
    isect_dense_region = n_isect * 0.35 / top_tiles
    print(f"    Estimated GS/tile in top 5% tiles: mean={np.mean(isect_dense_region):.0f}")
    
    # Active tiles = n_active_tiles, each has ~avg_isect_per_active
    # Fraction of tiles above threshold:
    avg = np.mean(avg_isect_per_active)
    if avg > 0:
        # Estimate using exponential distribution parameterized by mean
        frac_above = np.exp(-threshold / avg)
        n_above = n_active_tiles * frac_above
        print(f"    Estimated fraction of tiles above K={threshold}: {np.mean(frac_above)*100:.2f}%")
        print(f"    Estimated count: {np.mean(n_above):.0f} tiles ({np.mean(n_above)/n_tiles*100:.2f}% of all)")
        print(f"    Estimated intersection fraction: {np.mean(np.exp(-threshold / avg)):.6f}")
        
        if np.mean(frac_above) < 0.001:
            print(f"    → Effectively NO dense tiles at K={threshold}")
        elif np.mean(frac_above) < 0.01:
            print(f"    → Marginal dense tile population (<1%)")
        elif np.mean(frac_above) < 0.1:
            print(f"    → Moderate dense tile population ({np.mean(frac_above)*100:.1f}%)")
        else:
            print(f"    → Significant dense tile population")

# ─── Actually check macro-tile density (HiGS's own grouping) ───
print(f"\n--- HiGS Macro-Tile Density (each = 64 fine tiles) ---")
# With 135 macro-tiles for 1080p, average intersects per macro-tile:
avg_macro_isect = n_isect / n_macro
print(f"  Avg intersects per macro-tile: {np.mean(avg_macro_isect):.0f}")
print(f"  Min per macro-tile (uniform): {np.min(avg_macro_isect):.0f}")
print(f"  Max per macro-tile (uniform): {np.max(avg_macro_isect):.0f}")

# Since HiGS uses 1 warp per (macro-tile, batch) pair:
# batch = 128 GS per warp iteration
# Each macro-tile has avg_isect/128 batches
batches_per_macro = avg_macro_isect / 128
print(f"  Avg batches per macro-tile: {np.mean(batches_per_macro):.1f}")

# ─── Bimodality check ───
print(f"\n--- Bimodality Check ---")
# Using the CV to check if there's evidence of bimodality
# CV > 1 suggests heavy-tailed distribution
cv_isect = np.std(avg_isect_per_active) / np.mean(avg_isect_per_active) if np.mean(avg_isect_per_active) > 0 else 0
cv_vis = np.std(n_vis) / np.mean(n_vis) if np.mean(n_vis) > 0 else 0
cv_active = np.std(n_active_tiles) / np.mean(n_active_tiles) if np.mean(n_active_tiles) > 0 else 0
print(f"  CV (n_active_tiles): {cv_active:.3f}")
print(f"  CV (avg_isect_per_active): {cv_isect:.3f}")
print(f"  CV (n_visible): {cv_vis:.3f}")
print(f"  Note: CV < 1 = relatively uniform, CV > 1 = heavy-tailed")

# ─── Constant overhead analysis (from C33-D data) ───
print(f"\n--- Render Time Components ---")
print(f"  mean fwd_ms: {np.mean(fwd_ms):.2f}ms")
print(f"  mean total_render (fwd+bwd): {np.mean(total_render):.2f}ms")
print(f"  n_visible range: {np.min(n_vis):.0f} - {np.max(n_vis):.0f}")
print(f"  total_render range: {np.min(total_render):.2f}ms - {np.max(total_render):.2f}ms")
print(f"  total_render std: {np.std(total_render):.2f}ms ({np.std(total_render)/np.mean(total_render)*100:.1f}% of mean)")

# After iter 0, what's the render time stability?
post_warmup = [r for r in recs if r["step"] >= 10]
fwd_pw = np.array([r["fwd_ms"] for r in post_warmup])
total_pw = np.array([r["total_render_ms"] for r in post_warmup])
print(f"\n  Post-warmup (iter>=10):")
print(f"    fwd: mean={np.mean(fwd_pw):.2f}ms  std={np.std(fwd_pw):.2f}ms  CV={np.std(fwd_pw)/np.mean(fwd_pw):.3f}")
print(f"    total_render: mean={np.mean(total_pw):.2f}ms  std={np.std(total_pw):.2f}ms")
print(f"    min={np.min(total_pw):.2f}ms  max={np.max(total_pw):.2f}ms")
print(f"    (max-min)/mean = {(np.max(total_pw)-np.min(total_pw))/np.mean(total_pw)*100:.1f}% variation")

# ─── Key conclusion ───
print(f"\n=== C35-1 Conclusion ===")
print(f"n_active_tiles: {np.mean(n_active_tiles):.0f}/{n_tiles} ({np.mean(n_active_tiles)/n_tiles*100:.1f}%)")
print(f"Mean GS per active tile: {np.mean(avg_isect_per_active):.1f}")
print(f"Render time CV post-warmup: {np.std(total_pw)/np.mean(total_pw):.3f}")
print(f"→ If CV(total_render) < 0.1: render time is dominated by fixed overhead")
print(f"→ If CV > 0.3: render time is workload-sensitive (potential for adaptive blending)")
