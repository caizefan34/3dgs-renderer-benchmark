# MSB0-P0 Algebra — Candidate Derivations

All math in float64. Forward is front-to-back per pixel; buckets are groups of
`BUCKET = 32` Gaussians in depth order. Per bucket `b`, the forward stores the
**prefix** state at the bucket boundary:

- `C_b` = accumulated blended color just before bucket `b` (weighted sum of
  `T_j * alpha_j * color_j` over all Gaussians in buckets `< b`).
- `T_b` = transmittance entering bucket `b` = product of `(1 - alpha_j)` over all
  Gaussians in buckets `< b`.

The final pixel is `C_N + T_N * bg`, and `tile_final_transmittances = T_N`.

## Reverse-mode backward per bucket (M0 — faithful to `blend_backward_cu`)

Seeded from saved state `(C_b, T_b)` and `T_N`:

```
color_pixel_after = C_no_bg - C_b          # accumulated color AFTER bucket b
transmittance     = T_b                     # entering transmittance
grad_alpha_common = T_N * (-dot(grad_color_pixel, bg))
for j in bucket (reverse):
    blending_weight = transmittance * alpha_j
    dL_dcolor     += blending_weight * grad_color_pixel
    color_pixel_after -= blending_weight * color_j
    oma           = 1 - alpha_j
    oma_rcp       = 1 / max(oma, 1e-6)
    dL_dalpha      = dot(transmittance*color_j - color_pixel_after*oma_rcp, grad_color_pixel)
                     + grad_alpha_common * oma_rcp
    dL_dopacity    = gaussian_j * dL_dalpha
    ggh            = -alpha_j * dL_dalpha
    dL_dconic      = 0.5 * ggh * (dx^2, dx*dy, dy^2)     # kernel 0.5x cross-term convention
    dL_dmean2d     = ggh * (conic.x*dx + conic.y*dy, conic.y*dx + conic.z*dy)
    transmittance *= oma
```

**Key structural fact.** Within a bucket, the reverse walk needs only:
1. `T_b` (entering transmittance) — to seed `blending_weight` and the
   transmittance-recursion;
2. `C_b` (entering accumulated color) — to form `color_pixel_after`, which drives
   the `dL_dalpha` color term;
3. `T_N` (final transmittance) — for `grad_alpha_common`.

Each of these is **independent information**: `T_b` is a product over buckets
`< b`; `C_b` is a weighted color sum over buckets `< b`. Neither is a function of
the other or of bucket `b`'s own Gaussians.

## M1 — drop accumulated RGB (`C_b`), keep `T_b`

Set `C_b = 0` (cannot reconstruct). Then
`color_pixel_after = C_no_bg` (total accumulated color), which is wrong: the
correct value is `C_no_bg - C_b`. The `dL_dalpha` color term
`dot(transmittance*color - color_pixel_after*oma_rcp, grad_color_pixel)` is
therefore corrupted. `dL_dcolor` is **unaffected** (it needs only `T_b` via
`blending_weight`). Result: `dL_dalpha` and downstream `dL_dconic`/`dL_dmean2d`
collapse. **Not locally reconstructible.**

## M2 — drop transmittance (`T_b`), keep `C_b`

Set `T_b = 1`. Then `blending_weight = alpha_j` (missing the `transmittance`
factor) and the transmittance recursion never decays. `dL_dcolor` (needs `T_b`)
and `dL_dalpha` both collapse. `C_b` alone is insufficient. **Not locally
reconstructible.** (Also M2 removes only 4/16 = 25% of `bucket_color_transmittance`,
below the 25% Gate-3 bar once per-pixel buffers are counted.)

## M3 — replace saved state by forward-carry from known `(C_0=0, T_0=1)`

Reconstruct each bucket's entering state by carrying the exact forward recurrence:

```
carry_C = 0 ; carry_T = 1                       # entering bucket 0
for b in buckets (forward, sequential):
    C_b = carry_C ; T_b = carry_T              # exact entering state
    run reverse walk in bucket b using (C_b, T_b)
    # exiting state of bucket b = entering state of bucket b+1:
    carry_C = C_no_bg - color_pixel_after_after_bucket_b
    carry_T = transmittance_after_bucket_b
```

This is **bit-exact** (cos = 1.0, rel <= 2.2e-11 on all scenes) because it
re-derives the identical prefix the forward pass stored. But:

- It is a **second full tile-Gaussian traversal** (re-walk all Gaussians
  front-to-back) — violates Gate 5 "no second full traversal".
- The carry couples bucket `b+1`'s input to bucket `b`'s output, so buckets are
  **no longer independent** — the parallel per-bucket reverse walk is destroyed
  and a **global (cross-bucket) synchronization** is required — violates Gate 5
  "no global synchronization beyond existing boundaries".
- Mechanistically, M3 **is** the original pixel-centric 3DGS backward
  (reconstruct transmittance while traversing), i.e. prior-art adjacency.

## Minimality of `float4(C.rgb, T)`

`C_b` (3 floats) and `T_b` (1 float) are independently necessary (M1/M2 each
break a distinct gradient family). `float4` is therefore the minimal exact
replay summary; no smaller exact form exists. The only way to shrink it is to not
store it (M3), which reintroduces sequential work.
