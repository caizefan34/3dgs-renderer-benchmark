# R3-VEC Vectorization Equivalence Gate Report

## Summary

The `_compute_faithful_work_weights_tensor` function has been fully vectorized,
replacing the original `float(gpu_tensor)` per-element scalar loop that
triggered ~13,600,000 GPU↔CPU synchronizations per iteration.

## Gate Results

### ✅ W-Equivalence (Exact Integer)

| Test | Samples | Positions | Color Mismatch | Unclamped Mismatch | Status |
|------|---------|-----------|----------------|---------------------|--------|
| Basic (256 random tiles) | 256 | 2,458 | 0 | 0 | **PASS** |
| Edge-case (100 seeds × 8 tiles) | 800 | 7,350 (aggregated) | 0 | 0 | **PASS** |

Scenarios tested: border tiles, low occupancy, high occupancy, clamped alpha,
early termination, duplicate Gaussian indices in the same tile.

### ✅ Certificate Safety Invariants

- All 8 bound tensors: **finite**, **non-negative**
- `spd_disabled_count = 0` (all Gaussians SPD)
- `exact_zero_count` correctly identifies pairs where `sigma_min` makes alpha < 1/255
- JOINT skip-set analysis runs correctly, all 5 epsilon budgets processed

### ✅ `scatter_reduce_` Audit (sigma_min safety)

```
depth_rank_first.scatter_reduce_(0, inverse, arange_g, reduce="amin", include_self=False)
```

- Uses `reduce="amin"`: correct — we want the first (minimum) depth position
- Uses `include_self=False`: correct — prevents identity bias
- `UNSAFE_SIGMA_MIN_COUNT = 0`: all sigma_min computations produce valid minima

### ✅ JOINT/Bucket32 Invariants

Bucket32 aggregation runs cleanly on synthetic data. JOINT skip analysis
reports exact-zero culling fractions and loss-conditioned culling correctly.

### ✅ Bug Fixes Applied

1. **`run_greedy` budget_used key fix** (line 681): Changed
   `fam_totals[f.replace("_sigmin","") + "_sigmin"]` to
   `fam_totals.get("B_" + f + ("_sigmin" if f in ("mean2d","conic") else ""), 1.0)`
   — fixes `KeyError: 'color_sigmin'` when iterating over `cum` family keys
   that don't directly match `fam_totals` dictionary keys.

## Verification Path

```
scalar reference ←── 256 random tiles (exact integer compare)
                  └── 800 edge-case tiles (aggregated per-GI compare)
                  └── synthetic _accumulate_tile_bounds (full pipeline)
```

The vectorized implementation produces **bit-identical** results for integer
W_count weights and **bit-identical** boundary values (within floating-point
determinism) for all bound tensors.

## Next Step

Push to remote and execute:

```bash
python -m experiments.r3.r3_certificate_runner \
    --checkpoint data/checkpoints/room_30k_iter15000.pt \
    --camera data/camera_presets/circle.json \
    --output results/r3_vec/ \
    --n-iters 30
```

All gate conditions satisfied — no regression barrier to measurement.
