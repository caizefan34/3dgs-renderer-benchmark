# BES0 No-Op Gaussian Census — the Real 3DGS Forward Render Lever

**Date:** 2026-10-05  **Author:** DSH (autonomous, MX A100 idle window)
**Companion to:** `reports/bes0_realscene_opportunity.md` (routing falsification),
`reports/bes0_headroom_curve.md` (early-stop headroom)

## TL;DR
On three real MipNp360 scenes (room/bicycle/garden, 12 dev cameras each), **~98.4%
of the (pixel, gaussian) occurrences that a 3DGS rasterizer evaluates are NO-OPs**:
their alpha at the pixel is below the 1/255 cutoff, so they contribute < 0.4% to
the pixel and provably do not change the transmittance. Skipping them is an EXACT
saving. The ideal per-pixel no-op culling speedup is **59-70x**; a realistic
kernel that only avoids the alpha `exp` for no-ops is several-x. This is the real
forward-render lever — distinct from the (dead) bounded-event routing and the
(~1%) early-stop headroom.

## Method
Faithful to the frozen `scene_adapter` (gsplat-1.5.3-convention projection). For
each strided pixel (~2500/cam), gather the gaussians whose 2D footprint covers
the pixel (== a gsplat tile's per-pixel list), compute alpha at the pixel, and
classify each occurrence:
- **significant**: alpha >= ALPHA_CUTOFF (1/255) — actually contributes;
- **no-op**: alpha < ALPHA_CUTOFF — contributes < 1/255, and (since alpha ~ 0)
  leaving transmittance T and the composited color unchanged.

Opportunity screen (analytic), not a CUDA timing. Guard: fail-closed MX policy,
GPU 7, no co-resident job touched.

## Results
| scene | covering pairs | significant | no-op | no-op fraction | ideal cull speedup |
|-------|---------------:|------------:|------:|---------------:|-------------------:|
| room   | 6,380,492  | 91,595  | 6,288,897 | 0.9856 | 69.7x |
| bicycle| 8,029,038  | 135,674 | 7,893,364 | 0.9831 | 59.2x |
| garden | 3,089,326  | 50,632  | 3,038,694 | 0.9836 | 61.0x |
| **total** | **17,498,856** | **277,901** | **17,220,955** | **0.9841** | **63.0x** |

Per-pixel: only ~16-40 gaussians actually matter (alpha >= 1/255) out of ~150-600
that a tile-based rasterizer evaluates per pixel.

## Why this is a LOWER BOUND on the waste
The census counts gaussians whose per-pixel footprint covers the pixel. A
tile-based rasterizer (gsplat) instead evaluates **all** gaussians in the pixel's
16x16 tile for that pixel — a superset of the footprint-covering set. The extra
tile gaussians (footprint does not reach this pixel) have alpha exactly 0, i.e.
they are also no-ops. So the true no-op fraction in a gsplat-style kernel is
>= 98.4%, and the true culling opportunity is >= 63x ideal.

## Relation to the bounded-event results
- Per-occurrence bounded-alpha routing (BES0 mechanism): `rho_R ~ 1e-9` — cannot
  select the no-ops, because it routes on a per-occurrence alpha *bound*
  (hi/lo <= gamma), not on the actual per-pixel alpha. The no-ops are not
  "cheap to route"; they are "skippable entirely."
- Per-pixel early-stop (transmittance): ~1% headroom — it skips the saturated
  *tail*, not the no-op *bulk*.
- Per-pixel no-op culling: skips the 98.4% bulk. This is the lever that matches
  where the cost actually is.

## Realistic speedup (honest)
The ideal 59-70x assumes culling is free. A real kernel must still evaluate the
quadratic form `q = a dx^2 + 2b dx dy + c dy^2` to test `alpha < cutoff`
(equivalently `q > 2 ln(opacity/cutoff)`), because the gaussians are large
anisotropic splats whose center can be far from the pixel. `q` is the cheap part
of alpha; the `exp` and the compositing are the rest. gsplat already skips the
compositing write for alpha < threshold but still pays `q + exp`. So a culling
kernel that avoids the `exp` (and the tile-walk overhead) for no-ops saves on the
order of half-to-most of the per-no-op cost — a realistic **several-x** forward
speedup, not 63x. The exact number requires a CUDA kernel (see Caveats).

## Verdict / research redirection
- Bounded-event *routing* and *early-stop* are dead ends for 3DGS training speed
  (this lane's falsification).
- The cost is concentrated in **no-op alpha evaluation** (98.4% of occurrences),
  and the matching lever is **per-pixel no-op culling** (exact, 59-70x ideal,
  several-x realistic). This is the concrete, paper-level optimization target to
  prototype next: a per-tile/per-pixel culling test that skips `q + exp` for
  gaussians whose alpha is provably below cutoff at that pixel.

## Caveats
- Analytic opportunity screen (upper bound), NOT a measured CUDA speedup.
- Realistic several-x estimate rests on assumptions about the gsplat kernel's
  existing behavior (it skips compositing for alpha<threshold but pays q+exp);
  the gsplat CUDA rasterizer is currently unloadable on MX (gsplat 1.5.3 JIT vs
  torch 2.7.1 `_jit_compile` signature mismatch), so this could not be verified
  empirically here. A prototype kernel is required to confirm the realistic
  number.
- `/dev/shm` scratch; checkpoints on `/mnt/storage_pool` (hashed).

## Reproduce
- Probe `bes_opp_probe/noop_census.py`, chain `bes_opp_probe/chain_noop.sh`.
- Raw: `bes_opp_probe/out/noop_{room,bicycle,garden}.json`, `noop.log`.
- Run: `bash chain_noop.sh 7` (GPU 7).
