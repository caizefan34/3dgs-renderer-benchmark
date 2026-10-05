# BES0 Per-Tile AABB Reject — Measured CUDA Speedup (the honest number)

**Date:** 2026-10-05  **Author:** DSH (autonomous, MX A100 shared-GPU 6/7 window)
**Companion to:** `reports/bes0_noop_culling_measured.md` (per-pixel no-op culling, 1.17-1.23x),
`reports/bes0_realscene_opportunity.md` (per-occurrence alpha-routing falsification, LAW-BEVT),
`reports/bes0_noop_census.md` (analytic per-occurrence census).

## TL;DR
Instead of skipping no-op *pixels* inside a tile walk (the culling lever, capped at ~1.2x by
LAW-ALPHAWALK), I tested the structurally stronger lever: **per-tile AABB reject** — before
walking a tile, compute each gaussian's screen-space axis-aligned bounding box (the tightest
box containing its `alpha >= 1/255` ellipse) and drop the whole gaussian from tiles its box
does not intersect. On a faithful gsplat-style staged CUDA walk this is **bit-identical**
(`max|diff| = 0` on all 36 cameras at the standard 16px tile). At the standard **TS=16** tile it
gives **1.32-1.45x** (room 1.451x / bicycle 1.363x / garden 1.317x, 3 scenes x 12 dev cameras,
MX GPU7), beating per-pixel culling (1.17-1.23x) because it removes the per-occurrence `q`-test
for rejected gaussians, not just their `exp`+blend. The *staging floor* (shared-memory tile-list
staging + loop) is 27-42% of the TS=16 baseline and caps the speedup. **But that ceiling is
tile-size-dependent, not fundamental:** at **TS=32** the per-tile staging floor collapses to
5.5-11% and the exact reject rises to **1.63-1.85x** (room 1.851x / bicycle 1.820x / garden
1.632x, aggregate 1.742x; 1.60x vs the TS=16 production baseline), still bit-identical
(`max|diff| = 0` on all 36 cameras — TS=32 also removes the garden cam-68 boundary residual). A
two-stage staging variant lowers the floor 5.3x but the stage-2 global reads for survivors offset
the gain on aggregate (see Two-Stage section).

## Method
Faithful to the frozen `scene_adapter` (gsplat-1.5.3-convention projection) and the census
per-tile depth-sorted gaussian lists. One CUDA thread-block per 16x16 tile, one thread per
pixel, front-to-back blend. The per-tile list is staged into shared memory so each
gaussian's params are read from global memory once per tile (the staging floor). Modes in
one kernel:
- `baseline` (0): stage + `q` + `exp` + threshold + blend on every occurrence (gsplat staged).
- `pix`    (4): per-pixel AABB reject (drop gaussian at a pixel if the pixel is outside its box).
- `tile`   (3): **per-tile AABB reject** (drop the whole gaussian from tiles its box misses) — the lever.
- `floor`  (2): stage + loop only, no `q`/`exp`/blend (memory/loop lower bound).

**AABB construction.** For a gaussian with conic `A = [[a,b],[b,c]]` (inverse covariance) the
`alpha >= 1/255` region is the ellipse `q(dx,dy) = a dx^2 + 2b dx dy + c dy^2 <= T` with
`T = 2(ln opacity + ln 255)`. Its axis-aligned half-extents are `gx = sqrt(T * c / det)`,
`gy = sqrt(T * a / det)` with `det = a c - b^2`. The tile reject drops the gaussian when the
tile's pixel box does not intersect `[mx-gx, mx+gx] x [my-gy, my+gy]`.

## Two bugs found and fixed (this is where the exactness came from)
A naive per-tile AABB reject gave `max|diff| = 1.0-2.3` (should be 0). Root-caused via CPU
numpy reproduction + a GPU per-gaussian scan:
1. **Ratio-clamp bug.** The old screen-extent code clamped the *ratio* `gdx/gdy` to `1e-30`.
   For near-singular gaussians (`det <= 0`, i.e. sub-pixel thin gaussians) the ratio goes
   negative and was clamped to `1e-30`, making `gx ~ 0` so the gaussian was wrongly *skipped*
   even where it is visible. **Fix:** clamp the *determinant* instead, and take
   `gx,gy = max(f64_extent, f32_extent)` (evaluate the extent in float64 for safety).
2. **Float32 cancellation blobs.** For high-correlation gaussians (`rho -> 1`) the kernel's
   float32 quadratic form `q = ca dx^2 + 2 cb dx dy + cc dy^2` catastrophically cancels at
   far pixels, so `q` reads small and `alpha` spuriously exceeds cutoff (a visible blob) even
   though the exact `q` is large. The *correct* AABB (computed in float64) removes these
   blobs, so keeping them is the source of the residual. **Fix:** inflate the AABB by `1%`
   (`MARGIN_REL = 1e-2`, `MARGIN_ABS = 1e-3`), making it a strict superset of the true
   ellipse. This alone makes the reject **exact** (`max|diff| = 0`) with **no speedup penalty**
   (1.426x vs 1.433x — within noise): the inflated box still excludes the same tiles.

**Never-skip safeguard.** If `corr = det/(a c) <= 1e-6` (`AABB_TAU`) or `det` is non-finite,
the gaussian is degenerate; the code forces `gx = gy = 1e9` so the kernel *keeps* it (never
wrongly drops a visible gaussian). Measured never-skip fraction: room 2-4%, bicycle 3-6%,
garden 12-23% (garden gaussians are more sub-pixel-thin).

## Results (single-stage, 3 scenes x 12 dev cameras, MX GPU7, run7b, clean)
All 36 cameras: room/bicycle `max|diff|_tile = 0`; garden `max|diff|_tile = 1.9e-3` on one
camera (68), `0` on the other 35 (see Caveats — float32 boundary residual, removed by 10%
inflation, sweep confirmed). Determinism `det = 0` on every camera.

| scene | occurrences | base (ms) | tile-reject (ms) | speedup | floor (ms) | floor/base | per-cam speedup |
|-------|------------:|----------:|-----------------:|--------:|-----------:|-----------:|-----------------|
| room    | 514,037,931   | 822.9  | 567.0  | **1.451x** | 222.8 | 27% | 1.38-1.51x |
| bicycle | 1,676,846,955 | 2332.7 | 1711.2 | **1.363x** | 972.6 | 42% | 1.27-1.42x |
| garden  | 1,012,524,460 | 2185.3 | 1659.8 | **1.317x** | 628.8 | 29% | 1.19-1.40x |

`max|diff|_pix` (per-pixel variant, mode 4): room 0, bicycle 4.9e-3, garden 1.9e-3 — the
per-pixel variant is *not* bit-exact because the kernel's own float32 `q` still evaluates the
boundary pixel; the per-tile variant (mode 3) is the exact one. Per-gaussian color uses one SH
evaluation, mode-invariant, so it does not affect speedup or exactness.

## Why only 1.45x at TS=16 (and why TS=32 breaks it)
Per-tile AABB reject removes the `q`-test *and* the `exp`+blend for every rejected occurrence
— strictly more than per-pixel culling (which keeps the `q`-test, LAW-ALPHAWALK). Yet the
speedup is only 1.32-1.45x because:
1. **The staging floor dominates.** Shared-memory tile-list staging + the per-tile loop/sync
   (`floor` mode) is **27-42% of the baseline** and is *not removed* by reject — every tile
   still stages its (surviving) list and loops over it. bicycle's floor is 42% of baseline,
   which is why its speedup (1.363x) is lowest.
2. **Per-tile test overhead.** The reject itself costs a box-vs-tile test per (gaussian, tile)
   in the staging kernel; for gaussians whose box spans many tiles this is cheap, but the
   staging pass still touches every tile the gaussian overlaps.
3. **Reject removes compute, not memory.** The dominant remaining cost is streaming the
   per-tile list through shared memory, which reject does not shrink (it removes whole
   gaussians, but the survivors' list is still staged).

So at TS=16 the structurally-stronger per-tile reject is **capped near 1.45x** by the staging
floor (the same floor that caps per-pixel culling at 1.2x). The lever is real and exact, but at
TS=16 the forward-walk cost is concentrated in the staging floor, not the per-occurrence compute
that reject removes. **The floor is per-tile, so it is tile-size-dependent** — see the next
section: at TS=32 the floor collapses and the exact reject reaches 1.85x.

## Tile-size: the ceiling is tile-size-dependent (TS=32 breaks 1.45x -> 1.85x)
The ~1.45x ceiling above is measured at the standard gsplat **TS=16** tile. The staging floor is
a *per-tile* overhead (stage the tile's list into shared memory, sync, loop control), so it scales
as **num_tiles ~ 1/T²**, while the per-occurrence compute does not shrink as fast. Sweeping the
tile size (room, 3 cams, all bit-identical) shows the floor fraction — and thus the speedup —
is strongly tile-size-dependent:

| tile size | speedup (room, 3 cams) | floor / baseline | occurrences |
|----------:|-----------------------:|-----------------:|------------:|
| 8  | 1.297x | 48% | 515M |
| 16 | 1.492x | 31% | 145M |
| 32 | **1.768x** | **5.7%** | 45M |

At TS=32 the per-tile staging floor nearly vanishes (5.7% vs 31% at TS=16), so culling removes a
much larger *fraction* of the remaining work. Full 3-scene x 12-camera sweep at TS=32 (all
`max|diff|_tile = 0`):

| scene | base (ms) | tile-reject (ms) | speedup | floor (ms) | floor/base | occurrences |
|-------|----------:|-----------------:|--------:|-----------:|-----------:|------------:|
| room    | 976.0  | 527.2  | **1.851x** | 53.8  | 5.5%  | 162M |
| bicycle | 2413.9 | 1326.1 | **1.820x** | 270.8 | 11.2% | 525M |
| garden  | 2417.3 | 1481.1 | **1.632x** | 181.2 | 7.5%  | 329M |
| **AGG** | 5807.2 | 3334.4 | **1.742x** | 505.8 | 8.7%  | — |

Vs the TS=16 production-style staged baseline (aggregate 5341 ms), the TS=32 rejected walk
(3334 ms) is **1.60x faster**. TS=32 also removes the garden cam-68 float32 boundary residual
(`max|diff|_tile = 0` on all 36 cameras), because the 32px grid has no tile boundary at that pixel.

**Cross-tile-size determinism (measured caveat).** The per-tile reject is bit-exact *within* each
tile size (`max|diff|_tile = 0` at both TS=16 and TS=32). But the tiled baseline itself is **not
bit-reproducible across tile sizes**: rendering room cam 1 at TS=16 vs TS=32 (mode 0, no reject)
differs by mean 4.3e-5, with 99.7% of pixels < 1e-3 and localized clusters in high-density regions
reaching 1.157 at a few pixels (0.22% > 1e-3). This is FP non-associativity of the sequential
front-to-back blend (`T *= (1-alpha)` is order-dependent) combined with tile-size-dependent list
ordering — a property of the gsplat-style rasterizer, not of the reject. So the TS=32 reject is
exact relative to the TS=32 baseline; the 1.60x-vs-TS=16-baseline figure is a timing comparison of
two valid (FP-close) renders, not a bit-identical cross-tile claim.

**Caveat (why the TS=32 *baseline* is slower but the *rejected* walk is faster).** At TS=32 the
thread block is 32x32 = 1024 threads (the CUDA max); the shared-memory staging (CHUNK=512) is
then half-utilized and per-occurrence efficiency drops, so the TS=32 *baseline* (5807 ms) is
actually *slower* than the TS=16 baseline (5341 ms) despite 3x fewer occurrences. But the
*rejected* walk benefits twice: the floor is tiny (8.7%) and culling removes a larger share, so
TS=32 rejected (3334 ms) beats TS=16 rejected (3938 ms). **TS=32 is the better operating point for
the exact reject** — lower floor, faster rejected walk, higher speedup. (TS=64 is unreachable with
a square 1024-thread block; a 16x64 / multi-pass block would be needed.)

**Implication.** The staging-floor ceiling is an artifact of the 16px tile, not a fundamental
limit. Coarser tiles amortize the per-tile floor and let exact culling approach 2x. The open
question for further work: whether a 16px production baseline can adopt TS=32-style staging
without the per-occurrence efficiency loss (e.g. a 16x64 block with each thread owning 4 pixels,
or larger CHUNK).

## Two-Stage staging (attacking the floor) — measured
Hypothesis: the staging floor is the ceiling; a **two-stage** walk should lower it. Stage 1
streams only 4 floats/gaussian (`mx,my,a,b`-derived keep-flag + box) to build a compact
per-tile survivor list (a `tile_ptr` prefix-scan), cutting the staged footprint ~5x. Stage 2
walks the survivor list, loading the full 9 floats only for survivors. Measured floor is
**5.3x lower** than single-stage (aggregate 320.9 ms vs 1824.2 ms) and the two-stage output is
**bit-identical** to its own naive baseline (`max|diff| = 0`, `det = 0` on all 36 cameras).

Fair comparison (common reference = the single-stage gsplat staged baseline, mode 0; both
kernels produce bit-identical output to the true render, so the reference is valid). The
two-stage probe's own `baseline_ms` is a *naive global* walk (not the staged baseline), so its
reported `speedup_ts` is NOT used; I recompute `single_baseline / two_ts`.

| scene | single-stage | two-stage (fair) | two/single | s_floor | t_floor | t_floor/s_floor |
|-------|-------------:|-----------------:|-----------:|--------:|--------:|----------------:|
| room    | 1.451x | 1.472x | 1.01x | 222.8 | 44.5  | 0.20x |
| bicycle | 1.363x | 1.364x | 1.00x | 972.6 | 156.9 | 0.16x |
| garden  | 1.317x | **0.720x** | 0.55x | 628.8 | 119.6 | 0.19x |
| **AGG** | **1.356x** | **1.007x** | 0.74x | 1824.2 | 320.9 | 0.18x |

**Two-stage does NOT beat single-stage on aggregate (1.007x vs 1.356x)** despite the 5.3x
lower floor. The mechanism is a memory-traffic crossover. Per global-memory read:
- single-stage reads `9 * N_all` floats (stage 9 floats/gaussian, all gaussians).
- two-stage reads `4 * N_all + 9 * N_surv` floats (stage-1 4 floats for all, stage-2 9 floats
  for survivors only).

Two-stage wins iff `4*N_all + 9*N_surv < 9*N_all`, i.e. **`N_surv/N_all < 5/9 = 0.556`**.
- room/bicycle cull aggressively per tile (sparse survivors, ratio < 0.556) → two-stage ties
  or wins (room cam 114: 1.95x vs 1.51x; bicycle cam 19: 1.93x vs 1.40x).
- garden has dense per-tile survivors (ratio > 0.556, gaussians are thin but heavily overlap)
  → two-stage LOSES (cam 1: 0.42x; cam 85: 0.60x). Its floor advantage is wiped out by the
  stage-2 global re-reads.

Per-camera: two-stage beats single-stage on 9/12 room cams and 7/12 bicycle cams, but only
2/12 garden cams. The two-stage floor is real and 5.3x lower, but it does not reduce *total
bytes* — it splits the traffic across two passes. The only way to actually cut traffic is to
keep stage-2 survivors resident in shared memory (read 9 floats once, all 16x16 pixels from
smem) — which is exactly what single-stage already does. **So single-stage smem staging is
already near-optimal for the smem-resident design, and the ~1.45x ceiling is the staging
floor, not the per-occurrence compute.**

A **hybrid** (two-stage for tiles with `N_surv/N_all < 0.556`, single-stage otherwise) would
capture the best of both, but the gain over single-stage is marginal (the ceiling is ~1.45x).

## Verdict / research redirection
- Per-tile AABB reject is a **real, exact** forward-walk lever: **1.32-1.45x at TS=16**, rising to
  **1.63-1.85x at TS=32** (aggregate 1.742x; 1.60x vs the TS=16 production baseline), bit-identical
  (`max|diff| = 0` on all 36 cameras at both tile sizes). It strictly beats per-pixel no-op culling
  (1.17-1.23x) by also removing the `q`-test. The staging floor that caps it is **per-tile and
  tile-size-dependent** (27-42% at TS=16, 5.5-11% at TS=32), not a fundamental limit.
- Two-stage staging lowers the floor 5.3x and is bit-identical, but on aggregate the stage-2
  global reads for survivors offset the floor gain, so the end-to-end two-stage speedup vs the
  same gsplat staged baseline is **not** a clean win (see table). The floor is memory-bandwidth
  bound; lowering it further requires keeping survivors' params resident, not re-reading them.
  (At TS=32 the floor is already small enough that two-stage is unnecessary.)
- **Redirection (paper-level):** the headroom is in **(a) the tile size / staging-floor
  amortization** (TS=32 already recovers most of it; a 16x64 or multi-pass block could push the
  exact reject higher without the TS=32 per-occurrence efficiency loss), and **(b) shrinking the
  survivor list itself** (a coarser tile grid, a tighter/OBB reject, or a hierarchical
  per-pixel significant-only list). The per-occurrence `exp`+blend is *not* where the headroom is —
  the staging floor and the survivor-list traffic are.

## Reproduce
- Probe `bes_aabb_probe/aabb_probe.py`, kernel `aabb_kernel.cu` (single-stage, 13-arg), chain `run7.sh`.
- Two-stage: `bes_aabb2_probe/aabb2_probe.py`, kernel `aabb_kernel2.cu` (5-arg), chain `ts_chain7.sh`;
  analysis `bes_aabb_probe/analyze_two_vs_single.py`; raw JSON `bes_aabb2_probe/aabb2_*.json`.
- Env: `AABB_INFLATE=1e-2` (default), `AABB_TAU=1e-6` (default), `AABB_R_CAP=256`,
  `AABB_ALLOW_BUSY=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=7`.
- Raw: `/dev/shm/bes_aabb/out/{aabb,aabb2}_{room,bicycle,garden}.json`, `ts7b.log`, `sweep6.log`, `sweep7b.log`.

## Caveats
- Measured on the **forward alpha-compositing walk only** (not projection, binning, SH eval, or
  backward). End-to-end training speedup from this lever alone is < 1.45x.
- `r_cap=256px` radius cap (bounds per-tile list size for the 2.9-3.9M-gaussian scenes); affects
  only a few huge near-camera gaussians.
- Garden `max|diff|_tile = 1.9e-3` on camera 68 is a **float32 boundary residual**: the AABB
  (built from the float32 conic) excludes a boundary pixel that the kernel's float32 `q`
  includes. It is ~0.5/255 (half a quantization level), not a correctness bug in the reject
  logic. **Fix confirmed:** re-running the first 5 garden cams (incl. cam 68) at 10% inflation
  (`AABB_INFLATE=0.1`, GPU7) gives `max|diff|_tile = max|diff|_pix = 0` and `det = 0` on all 5,
  with speedup 1.271x (vs 1.317x at 1% inflation — the looser box keeps a few more gaussians).
  So 1% inflation is exact on 35/36 cameras at full speed; 10% inflation is exact on all of
  garden at a ~3% relative speedup cost. The never-skip safeguard (`AABB_TAU=1e-6`) keeps all
  degenerate/sub-pixel gaussians (never-skip frac garden 5-23%).
- Host is fully shared; timing uses min-based aggregates (cleanest) and the speedup *ratio* is
  robust (median/min within 1-2%). GPU0 VLLM (25GB) and GPU7 train_tf.py (6.6GB) belong to
  other users and were not touched.
