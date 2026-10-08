# DWC0-P0 Report — Duplicate-Work Check: Shared Gaussian Computation

**Date (execution):** 2026-10-01 (Asia/Shanghai)
**Repo:** /root/3dgs-renderer-benchmark @ `abfa6c3e4b23caf5a927bf3722a65cba37b6fab3` (branch `r6-exact-backward-screening`)
**Scope:** THEORY / SOURCE / TRACE / CPU-ORACLE ONLY. **No CUDA built or run.**

## Core question

Can shared Gaussian computation reduce duplicated work in the tile rasterizer? Specifically, are projection, covariance, SH evaluation, or tile assignment recomputed per tile/pixel, so that computing them once and reusing would cut duplicate compute?

## Answer

**No.** The canonical tile rasterizer **already** computes projection, covariance (3D+2D), and SH **exactly once per Gaussian** and reuses them across every tile and pixel the Gaussian touches. The only per-(pixel,gaussian) recompute is the Gaussian weight `exp(-sigma)`, a pure function of the per-pixel offset — irreducible. There is no duplicated heavy compute to eliminate.

## Gate results

| Gate | Description | Verdict |
|------|-------------|---------|
| 1 | Duplicate-computation reduction ≥ 20% | **FAIL** — reduction vs current = 0% |
| 2 | Memory overhead ≤ 10% | **PASS** — 0% new storage (candidate == current) |
| 3 | Amdahl upper bound ≥ 1.1× | **FAIL** — bound = 1.0× |

**Verdict: `DROP_NO_DUPLICATE_WORK_HEADROOM`** (first failing gate = Gate 1).

## Phase 0 — Source lock

Frozen the four target computations in both the INRIA reference backend (frozen C0) and the gsplat default fused path:

| Quantity | Where computed | Instances |
|---|---|---|
| projection | `preprocessCUDA` forward.cu:204-239 (`transformPoint4x4`+`ndc2Pix`); gsplat `ProjectionEWA3DGSFused.cu` | **1 / Gaussian** |
| covariance 3D+2D | `computeCov3D` forward.cu:217, `computeCov2D` forward.cu:222 | **1 / Gaussian** |
| SH evaluation | `computeColorFromSH` forward.cu:249; gsplat `SphericalHarmonicsCUDA.cu` | **1 / Gaussian** |
| tile assignment | `getRect`+`tiles_touched` forward.cu:241-262; `duplicateWithKeys` rasterizer_impl.cu:120; gsplat `IntersectTile.cu` | **1 / Gaussian** (+ index-only duplication) |

`renderCUDA` (forward.cu:269) reads the precomputed `points_xy_image[]`, `conic_opacity[]`, `features[]` and recomputes **only** the pixel-dependent weight:
```
d     = xy - pixel_center
power = -0.5*(conic.x*d.x^2 + conic.z*d.y^2) - conic.y*d.x*d.y
alpha = min(0.99, opacity*exp(power))
```
`d` is distinct per pixel, so `exp(-sigma)` cannot be shared across pixels.

The same holds for gsplat's **alternative fused from-world path** (`RasterizeToPixelsFromWorld3DGS.cuh`): `iscl_rot` (=R·inv(S), the covariance analog) and `scale` are staged **once per Gaussian** into shared/registers; the per-(pixel,gaussian) loop then does only the ray-splat evaluation `iscl_rot*(ray_o[p]-xyz)`, `grayDist`, `alpha=opac*__expf(-0.5*grayDist)`. Its inputs `ray_o[p]`/`ray_d[p]` are per-pixel, so that work is likewise irreducible and is not projection/cov/SH recompute.

## Phase 1 — Duplication trace (real anchors)

Real per-scene anchors: BARS0 real tile trace (iter 25000, 1920×1080, seed 42), PSB0 per-pixel visit totals, H6 intersection accounting. `n_tiles = 8160` (120×68 @ 16×16).

| Scene | N (30k) | mean splats/tile | tiles/gauss | pixel visits | **proj ratio** | **SH ratio** |
|---|---|---|---|---|---|---|
| room | 1,105,873 | 234.69 | 1.73 | 14,894,848 | **1.0** | **1.0** |
| garden | 874,019 | 580.89 | 5.42 | 35,872,768 | **1.0** | **1.0** |
| bicycle | 2,589,484 | 399.31 | 1.26 | 25,459,968 | **1.0** | **1.0** |

`duplicate_compute_ratio = total_compute_instances / unique_gaussian_computation = 1.0` for projection, covariance, and SH in every scene — the compute happens once; the higher `reuse_read_count` (tiles touched / pixels affected) is **read-only reuse**, not recompute.

## Phase 2 — Cost model

FLOP weights (FMA-equiv): projection ≈ 40, cov3d+cov2d+conic ≈ 260, SH ≈ 150 → eager bundle ≈ 450 flop/gaussian (once); Gaussian weight ≈ 20 flop + 1 exp per (pixel,gaussian) visit.

- **Current (canonical eager):** `N·450 + V·(20+8_exp)` — projection/cov/SH once, weight per visit.
- **Candidate "compute once + reuse":** **identical to current** — the canonical design already is compute-once+reuse.
- **Counterfactual per-visit** (recompute proj+cov+SH every visit): `V·(450+28)` — large, but **does not apply**, because the rasterizer does not do this.
- **Reduction vs real current = 0%**, new storage = 0% (eager buffers `points_xy_image`/`conic_opacity`/`features` already exist), sync-neutral, cache pressure lower in the eager design.

## Phase 3 — Amdahl upper bound

`speedup = 1/(1 − f_removed)`. Because projection/cov/SH are already eager (not duplicated) and the sole per-visit work (weight `exp`) is irreducible, `f_removed = 0` → **Amdahl bound = 1.0×**, far below the 1.1× bar. Even a delusional bound that falsely assumes proj+cov/SH are recomputed per visit AND that raster is the whole pipeline barely exceeds 1.0× (see `DWC0_AMDAHL.json`).

## Verdict

**`DROP_NO_DUPLICATE_WORK_HEADROOM`**

The shared-Gaussian-computation hypothesis is **already realized** in the canonical rasterizer: projection, covariance, and SH are computed once per Gaussian and reused everywhere. `duplicate_compute_ratio = 1.0` (no duplication), real reduction = 0%, memory overhead = 0%, Amdahl bound = 1.0×. Gate 1 fails first → DROP.

## What would change the verdict

- A **per-Gaussian** cost reduction (not duplicated-work elimination): cheaper `computeCov2D` mat3 chain, lower SH degree, or a fused cov2d+SH kernel. This reduces the eager bundle, it does not remove duplication.
- An **approximation** of the per-pixel Gaussian weight (e.g., box-sum / integral-image footprint) that breaks exactness — out of scope for an exactness-preserving direction and not "shared computation."

## Reproduce

```
python3 dwc0_headroom/scripts/dwc0_oracle.py
```

Artifacts: `DWC0_VERDICT.json`, `DWC0_SOURCE_LOCK.json`, `DWC0_DUPLICATION_STATS.json`, `DWC0_COST_MODEL.json`, `DWC0_AMDAHL.json`, `DWC0_REPORT.md`, `scripts/dwc0_oracle.py`, `MANIFEST.sha256`.
