# Reality gate — real projected-data availability

Preregistered data-availability search, CPU-only, read-only. Result: **DATA FOUND** (not
DATA_UNAVAILABLE). No synthetic substitution performed.

## What was searched
ripgrep/glob across the repo for the required schema (means2d, conics [a,b,c], opacities,
colors, depths, height, width) in .npz/.npy/.pt and in loader/export scripts. Rejected
candidates with reasons (synthetic assets with torch.rand/randn attributes, gradient/sketch
vectors, densification traces, bare checkpoints, remote-only `/home/...` paths not present
locally, tomography probe data).

## Confirmed authentic projected data

### Tier 1 (best) — native_tiles_*.npz family
- Path: `experiments/training_solver_design/TILE_PREFIX_TRANSMITTANCE_GATE_20260928/`
- 8 files, one per training camera (DSCF4668 … DSCF4977), room scene.
- Renderer: FastGS / INRIA `diff-gaussian-rasterization` (CUDA), iteration-30000
  point_cloud.ply (572,546 Gaussians).
- Arrays: projected_xy (means2d), conic_opacity (conics a,b,c + opacity), rgb (colors),
  depth, plus real kernel binning state (gaussian_ids, offsets, tile_ids, sorted_keys,
  native_n_contrib, native_final_T).
- Frame: 1600×1060, tile 16, tile grid 100×67, mult=0.5 (authoritative, from
  camera_tile_manifest.json sidecar).
- SHA256: see COST_MODEL.json `real_cohort.native_tiles_sha256`.

### Tier 2 — bfc0_p0/output/trace/room_trace.npz
- Room, gsplat rasterization, 3114×2075, N=100000, has width/height keys but NO depths
  (dropped at save). Conics ~99.76% zero-padded for culled Gaussians.

### Tier 3 — cgt_f0_fused_cuda/results/workload.npz
- Room + garden + bicycle, gsplat, but tile-local coords, no depths, no H/W key, subsampled.

## Caveat relevant to G3
The shipped cpu_oracle.validate() hard-asserts SPD conics. In Tier-2 room_trace.npz only
0.24% of conics are SPD (the rest zero-padded by gsplat for off-screen/culled Gaussians);
Tier-1 native_tiles has ~35–50% SPD (rest culled by the kernel's radius/opacity test).
To run the oracle on real data one must first restrict to the SPD+visible subset (exactly
what the rasterizer does). This is legitimate culling, not data faking — but the oracle
lacks a dense fallback for genuinely non-SPD/near-singular conics, so the preregistered
near-singular robustness scenario (G3) is not testable with the shipped code as-is.
