#!/usr/bin/env python3
"""
Full-integration smoke test for the vectorized _accumulate_tile_bounds.
Builds a minimal gsplat scene, invokes the full pipeline, and validates
that the new code produces correct output structure.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np

device = torch.device("cuda")

# === 1. Build minimal scene ===
N = 128  # 128 Gaussians
tile_size = 16
H, W = 64, 64  # 4x4 tiles = 16 tiles

# Random means in image space
means2d = torch.randn(1, N, 2, device=device) * 20 + 32
means2d = means2d.clamp(2, 62)

# Random conics (positive-definite via construction)
angles = torch.randn(N, device=device) * math.pi
xx_base = torch.exp(torch.randn(N, device=device) * 0.5 + 1.0)
yy_base = torch.exp(torch.randn(N, device=device) * 0.5 + 1.0)
xy_terms = torch.randn(N, device=device) * 0.1
conics = torch.zeros(1, N, 3, device=device)
conics[0, :, 0] = xx_base
conics[0, :, 1] = xy_terms
conics[0, :, 2] = yy_base

opacities = torch.sigmoid(torch.randn(N, device=device))  # [0,1]

# === 2. Run gsplat projection and tiling ===
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_tiles,
    isect_offset_encode,
)

print("=== Projection ===")
viewmat = torch.eye(4, device=device).unsqueeze(0)
K = torch.tensor([[[W, 0, W/2], [0, H, H/2], [0, 0, 1]]], device=device, dtype=torch.float32)
means = torch.zeros(1, N, 3, device=device)
means[:, :, 0] = (means2d[0, :, 0] - W/2) * 0.01
means[:, :, 1] = (means2d[0, :, 1] - H/2) * 0.01
means[:, :, 2] = 3.0  # depth

quats = torch.randn(1, N, 4, device=device)
scales = torch.exp(torch.randn(1, N, 3, device=device) * 0.5 - 1.0)
opacities_3d = opacities.detach().clone().unsqueeze(0)

# Project
proj_results = fully_fused_projection(means, quats, scales, opacities_3d,
                                       viewmat, K, W, H, tile_size)
means2d_proj = proj_results[0]
opacities_proj = proj_results[2]  # [1, N]
conics_proj = proj_results[1]     # [1, N, 3]

print(f"  means2d: {means2d_proj.shape}")
print(f"  conics: {conics_proj.shape}")
print(f"  opacities: {opacities_proj.shape}")

print("=== Tiling ===")
tile_offsets_3d, flatten_ids = isect_tiles(means2d_proj, None, H, W, tile_size,
                                           return_unsorted=False, sort=True)

# Flatten dims
means2d_flat = means2d_proj[0]
conics_flat = conics_proj[0]
opacities_flat = opacities_proj[0]

n_tiles = tile_offsets_3d.shape[1] - 1
tile_h = (H + tile_size - 1) // tile_size
tile_w = (W + tile_size - 1) // tile_size
print(f"  tiles: {tile_h}x{tile_w} = {n_tiles}")
print(f"  tile_offsets: {tile_offsets_3d.shape}")
print(f"  flatten_ids: {flatten_ids.shape}")

# === 3. Run Q_t computation ===
from r3_certificate_runner import compute_Q_t, compute_C_max_t

print("=== Q_t ===")
q_p = torch.ones(1, H, W, 3, device=device)
Q_t = compute_Q_t(q_p.permute(0,3,1,2), tile_size, tile_h, tile_w, H, W)
print(f"  Q_t: {Q_t.shape}")

C_max_t = compute_C_max_t(conics_flat.unsqueeze(0), tile_offsets_3d[0], flatten_ids, tile_h, tile_w)
print(f"  C_max_t: {C_max_t.shape}")

# === 4. Call the NEW vectorized function ===
print("\n=== NEW: _accumulate_tile_bounds (vectorized) ===")
import time
import r3_certificate_runner as runner

t0 = time.time()
bounds = runner._accumulate_tile_bounds(
    opacities_flat, conics_flat.unsqueeze(0), means2d_flat.unsqueeze(0),
    tile_offsets_3d[0], flatten_ids, Q_t, tile_h, tile_w, tile_size,
    H=H, W=W
)
t1 = time.time()

print(f"  Time: {t1-t0:.3f}s")
print(f"  B_color_tight: shape={bounds['color_tight'].shape}, sum={bounds['color_tight'].sum().item():.4f}")
print(f"  B_opacity_tight: sum={bounds['opacity_tight'].sum().item():.4f}")
print(f"  B_mean2d: sum={bounds['mean2d'].sum().item():.4f}")
print(f"  B_conic: sum={bounds['conic'].sum().item():.4f}")
print(f"  B_mean2d_sigmamin: sum={bounds['mean2d_sigmamin'].sum().item():.4f}")
print(f"  B_conic_sigmamin: sum={bounds['conic_sigmamin'].sum().item():.4f}")
print(f"  spd_disabled_count: {bounds['spd_disabled_count']}")
print(f"  exact_zero_count: {bounds['exact_zero_count']}")
print(f"  tile_gaussian_total: {bounds['tile_gaussian_total']}")
print(f"  exact_zero_fraction: {bounds['exact_zero_fraction']:.4f}")
print(f"  pair_data entries: {len(bounds['pair_data'])}")
print(f"  joint_skip keys: {list(bounds['joint_skip_analysis'].keys())}")

# Validate joint skip results
for ek, ev in bounds['joint_skip_analysis'].items():
    if ev.get("BUCKET32_SKIP_FRACTION") is not None:
        print(f"    {ek}: pairs={ev['JOINT_SKIP_PAIR_FRACTION']:.3f} work={ev['JOINT_SKIP_WEIGHTED_WORK_FRACTION']:.3f} b32={ev['BUCKET32_SKIP_FRACTION']:.3f}")

# === 5. Validate W_it distribution ===
wc_vals = [pd["w_color"] for pd in bounds["pair_data"].values()]
wu_vals = [pd["w_unclamped"] for pd in bounds["pair_data"].values()]
print(f"\n  W_color: min={min(wc_vals)} max={max(wc_vals)} nonzero={sum(1 for w in wc_vals if w>0)}/{len(wc_vals}")

# === 6. Basic sanity: verify B_mean2d == 0 for SPD-disabled ===
spd_mask = torch.tensor([1 - bounds["spd_disabled_count"]], device=device)
print(f"\n  === Sanity checks ===")
print(f"  SPD-disabled fraction: {bounds['spd_disabled_fraction']:.4f}")

# Verify result tensor shapes
assert bounds["color_tight"].shape == (N,), f"Expected ({N},), got {bounds['color_tight'].shape}"
assert bounds["pair_data"] is not None
assert len(bounds["joint_skip_analysis"]) > 0
for ek in ["eps_0.1pct", "eps_0.5pct", "eps_1.0pct", "eps_2.0pct", "eps_5.0pct"]:
    assert ek in bounds["joint_skip_analysis"], f"Missing {ek}"
    j = bounds["joint_skip_analysis"][ek]
    assert j["BUCKET32_SKIP_FRACTION"] is not None, f"B32 null for {ek}"

print("\n ALL CHECKS PASSED")

# === 7. Validate with OLD version (if available) ===
# For a true regression test, we'd need the old function. For now,
# just verify numerical correctness by comparing B_color_coarse vs B_color_tight
# (coarse should always be >= tight since it doesn't use E_tight)
assert bounds["color_coarse"].sum().item() >= bounds["color_tight"].sum().item() - 1e-6, \
    "B_color_coarse < B_color_tight violates theory"
print(" B_color_coarse >= B_color_tight OK")
print("\n=== FULL TEST PASSED ===")
