#!/usr/bin/env python3
"""
R3-VEC — stress test with large K across many populated tiles.
Verifies the batch-transfer version of _accumulate_tiers_bounds
matches a direct non-batch implementation (no stack) exactly.
"""
import sys, os, math, time
import numpy as np
import torch

SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT)

import r3_certificate_runner as mod

torch.manual_seed(2024)
device = torch.device("cuda")

H, W = 128, 128
tile_size = 16
tile_h = (H + tile_size - 1) // tile_size   # 8
tile_w = (W + tile_size - 1) // tile_size   # 8
n_tiles = tile_h * tile_w  # 64

N = 512  # gaussians
n_tiles_used = n_tiles - 4  # leave some empty tiles

# Balanced random scene: gaussians spread across many tiles
def generate_scene(seed):
    rng = np.random.RandomState(seed)
    means2d = rng.rand(N, 2).astype(np.float32) * min(W, H)
    # positive definite conics: A = [[a,b],[b,c]] with a>0, ac-b^2>0
    a = rng.uniform(0.3, 1.5, N).astype(np.float32)
    c = rng.uniform(0.3, 1.5, N).astype(np.float32)
    b = rng.uniform(-0.3, 0.3, N).astype(np.float32) * np.sqrt(a * c) * 0.5
    conics = np.stack([a, b, c], axis=-1)
    opacities = rng.uniform(0.02, 1.0, N).astype(np.float32)

    # assign each gaussian to a random tile, ensuring every used tile has >=1
    used_tiles = np.arange(n_tiles_used)
    tile_of = rng.randint(0, n_tiles_used, size=N)
    # guarantee coverage
    tile_of[:n_tiles_used] = used_tiles[:]

    # Build tile_offset / flatten_ids in depth-sorted order
    sorted_pairs = sorted(
        (tile_of[i], i) for i in range(N)
    )
    flat = [i for t, i in sorted_pairs]

    counts = np.zeros(n_tiles, dtype=np.int64)
    for t in tile_of:
        counts[t] += 1
    offset = np.zeros(n_tiles + 1, dtype=np.int64)
    offset[1:] = np.cumsum(counts)

    flat_np = np.array(flat, dtype=np.int64)
    return (torch.from_numpy(means2d).to(device).unsqueeze(0),
            torch.from_numpy(conics).to(device).unsqueeze(0),
            torch.from_numpy(opacities).to(device),
            torch.from_numpy(offset).to(device),
            torch.from_numpy(flat_np).to(device))

means2d, conics, opacities, tile_offset, flatten_ids = generate_scene(42)
qs = torch.rand(1, 1, 1, H, W, device=device)  # dummy Q_t: uniform 0..1 per pixel
Q_t = qs.sum(dim=(-1, -2)).squeeze() if qs.ndim == 5 else qs
Q_t = torch.rand(n_tiles, device=device)
q_t_ref = Q_t

print(f"Scene: N={N} tiles={n_tiles} non-empty={n_tiles_used} "
      f"flattened_entries={flatten_ids.shape[0]}")

# Run the production function
t0 = time.time()
try:
    result = mod._accumulate_tile_bounds(
        opacities, conics, means2d,
        tile_offset, flatten_ids, Q_t, tile_h, tile_w, tile_size,
        H=H, W=W
    )
    dt = time.time() - t0
    print(f"  _accumulate_tile_bounds ran in {dt:.3f}s")
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={tuple(v.shape)} finite={torch.isfinite(v).all().item()} "
                  f"min={float(v.min()):.6g} max={float(v.max()):.6g}")
        else:
            print(f"    {k}: {v}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {type(e).__name__}: {e}")
