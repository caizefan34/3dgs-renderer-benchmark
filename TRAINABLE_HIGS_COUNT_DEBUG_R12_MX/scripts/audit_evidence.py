#!/usr/bin/env python3
"""R12 P0: Audit R10/R11 evidence"""
import json

print("=== R12 P0: R10/R11 Evidence Audit ===\n")

# R11 claimed metadata
r11_metadata = {
    'n_gaussians': 580000,
    'n_visible': 273234,
    'n_isects': 490387290,
    'culling_ratio': 0.5289068965517241
}

print("1. R11 Metadata Source:")
print(f"   n_gaussians: {r11_metadata['n_gaussians']:,}")
print(f"   n_visible: {r11_metadata['n_visible']:,}")
print(f"   n_isects: {r11_metadata['n_isects']:,}")
print(f"   culling_ratio: {r11_metadata['culling_ratio']:.4f}")
print()

# Verify culling
expected_visible = int(r11_metadata['n_gaussians'] * (1 - r11_metadata['culling_ratio']))
print(f"2. Culling verification:")
print(f"   Expected visible: {expected_visible:,}")
print(f"   Actual visible: {r11_metadata['n_visible']:,}")
print(f"   Match: {expected_visible == r11_metadata['n_visible']}")
print()

# Calculate tile grid
width, height = 960, 540
tile_size = 16
tile_width = (width + tile_size - 1) // tile_size
tile_height = (height + tile_size - 1) // tile_size
n_tiles = tile_width * tile_height

print(f"3. Tile grid:")
print(f"   Image: {width}×{height}")
print(f"   Tile size: {tile_size}×{tile_size}")
print(f"   Tile grid: {tile_width}×{tile_height} = {n_tiles:,} tiles")
print()

# Physical bounds
print(f"4. Physical bounds:")
print(f"   Max tiles per Gaussian-camera pair: {n_tiles}")
print(f"   Max intersections (visible × tiles): {r11_metadata['n_visible'] * n_tiles:,}")
print(f"   Actual n_isects: {r11_metadata['n_isects']:,}")
print(f"   Ratio: {r11_metadata['n_isects'] / (r11_metadata['n_visible'] * n_tiles):.2f}x")
print()

# Per-Gaussian coverage
avg_visible = r11_metadata['n_isects'] / r11_metadata['n_visible']
avg_total = r11_metadata['n_isects'] / r11_metadata['n_gaussians']

print(f"5. Per-Gaussian coverage:")
print(f"   Per visible Gaussian: {avg_visible:.2f} tiles")
print(f"   Per total Gaussian: {avg_total:.2f} tiles")
print(f"   Expected: ~9 tiles (typical 3DGS)")
print(f"   Over-count: {avg_visible / 9:.1f}x")
print()

# R10 OOM analysis
print(f"6. R10 OOM allocation analysis:")
oom_bytes = 119488359424
print(f"   OOM bytes: {oom_bytes:,}")
print(f"   = {oom_bytes / 1e9:.2f} GB")
print(f"   = {oom_bytes / 1024**3:.2f} GiB")
print()

# Single int64 array
int64_elements = oom_bytes // 8
print(f"   If single int64 array:")
print(f"     Elements: {int64_elements:,}")
print(f"     = {int64_elements / 1e9:.2f} B elements")
print()

# Two arrays
two_array_elements = oom_bytes // 12
print(f"   If two arrays (int64 + int32):")
print(f"     Elements: {two_array_elements:,}")
print(f"     = {two_array_elements / 1e9:.2f} B elements")
print()

print(f"7. Conclusion:")
print(f"   R10 stack trace shows at::empty at IntersectTile.cu:833")
print(f"   Line 833: isect_ids = at::empty({{n_isects}}, opt.dtype(at::kLong))")
print(f"   This is SINGLE int64 array allocation")
print(f"   Expected n_isects from OOM: {int64_elements:,}")
print(f"   R11 claimed n_isects: {r11_metadata['n_isects']:,}")
print(f"   Ratio: {int64_elements / r11_metadata['n_isects']:.2f}x")
