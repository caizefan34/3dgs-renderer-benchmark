# MSB0-P0 Dependency Graph — Saved Checkpoint Components

For each component saved by the FasterGSFused forward pass, we classify whether
the backward gradient depends on it as **independent information** or whether it
is **exactly reconstructible** from other already-available quantities
(final pixel outputs, final transmittance, current-bucket Gaussian
alpha/color, neighboring bucket summaries, or already-required state).

## Saved components (per render, full image)

| Component | Buffer | Granularity | Bytes | Backward role |
|-----------|--------|-------------|-------|---------------|
| `bucket_color_transmittance` | `BucketBuffers.color_transmittance` | float4 per (bucket, tile-pixel) | 16 B | seeds each bucket's reverse walk with `(C_b, T_b)` |
| `tile_final_transmittances` | `TileBuffers.final_transmittances` | float per pixel | 4 B | `T_N` for `grad_alpha_common` |
| `tile_n_processed` | `TileBuffers.n_processed` | uint per pixel | 4 B | `last_contributor` (which Gaussians to process) |

## Component analysis

### `bucket_color_transmittance` = `float4(C_b.rgb, T_b)` — SPLIT into two scalars

**`C_b` (prefix accumulated color, 3 floats) — INDEPENDENT (not reconstructible)**
- Consumed by: `color_pixel_after = C_no_bg - C_b`, which drives the
  `dL_dalpha` color term `dot(transmittance*color - color_pixel_after*oma_rcp, gC)`.
- Reconstructing `C_b` requires the weighted color sum over ALL Gaussians in
  buckets `< b` — a cross-bucket prefix scan. Not available from final outputs,
  current-bucket gaussians, or `T_b`.
- Evidence: **M1** (drop `C_b`, set 0) → `dL_dalpha` cos = 0.0197 (room),
  0.839 (garden), 0.0735 (bicycle); `dL_dconic`/`dL_dmean2d` also collapse.
  `dL_dcolor` preserved (cos = 1.0) — confirming `C_b` is needed ONLY for the
  alpha path, and is not locally derivable.

**`T_b` (prefix transmittance, 1 float) — INDEPENDENT (not reconstructible)**
- Consumed by: `blending_weight = transmittance * alpha` (drives `dL_dcolor`)
  and the transmittance recursion `transmittance *= (1 - alpha)` (drives
  `dL_dalpha`).
- Reconstructing `T_b` requires the product of `(1 - alpha_j)` over all Gaussians
  in buckets `< b` — again a cross-bucket prefix scan.
- Evidence: **M2** (drop `T_b`, set 1) → `dL_dcolor` cos = 0.4498 (room),
  0.6821 (garden), 0.2452 (bicycle); `dL_dalpha` cos = 0.056/0.061/0.063. Both
  collapse. `C_b` alone is insufficient.

### `tile_final_transmittances` = `T_N` — INDEPENDENT (per-pixel, tiny)
- Consumed by: `grad_alpha_common = T_N * (-dot(gC, bg))`.
- `T_N` is the product over ALL Gaussians in the pixel (not just one bucket);
  not reconstructible from any single bucket's state. 4 B/pixel — negligible.

### `tile_n_processed` — INDEPENDENT (per-pixel, tiny)
- Consumed by: `last_contributor` skip logic (which Gaussians contributed).
- Not reconstructible without re-running the forward. 4 B/pixel — negligible.

## Reconstructibility summary

| Component | Independent? | Exact reconstruction path | Gate |
|-----------|--------------|---------------------------|------|
| `C_b` (RGB) | YES | only via M3 sequential forward carry | Gate 4 fail for M1 |
| `T_b` (T) | YES | only via M3 sequential forward carry | Gate 4 fail for M2 |
| `T_N` | YES | none cheaper than storing | retained |
| `n_processed` | YES | none cheaper than storing | retained |

## Conclusion

The dominant saved state `bucket_color_transmittance = float4(C_b, T_b)` carries
**two independent, non-reconstructible scalars** per (bucket, pixel). The float4
is the minimal exact replay summary. Elimination of either component breaks a
distinct gradient family (Gate 4). Joint elimination is exact only via M3's
sequential forward carry, which is a second traversal + global sync (Gate 5) and
is the original pixel-centric 3DGS backward (prior art).
