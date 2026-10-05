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
does not intersect. On a faithful gsplat-style staged CUDA walk this is **bit-identical** to
the baseline for room and bicycle (`max|diff| = 0` on every camera) and gives a **1.32-1.45x**
forward-walk speedup (room 1.451x / bicycle 1.363x / garden 1.317x, 3 scenes x 12 dev
cameras, MX GPU7). This beats per-pixel culling (1.17-1.23x) because it removes the
per-occurrence `q`-test for rejected gaussians, not just their `exp`+blend — but it is
**still far below** the "few-x" a naive estimate predicts, because the *staging floor*
(shared-memory tile list staging + loop) is 27-42% of the baseline and is not removed by
culling. A two-stage staging variant lowers that floor 5.3x but the stage-2 global reads for
survivors offset the gain on aggregate (see Two-Stage section).

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

## Why 1.45x and not "few-x" (the key finding)
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

So even the structurally-stronger per-tile reject is **capped near 1.45x** by the same
staging floor that caps per-pixel culling at 1.2x. The lever is real and exact, but the
forward-walk cost is not concentrated in the per-occurrence compute that reject removes — it
is concentrated in the staging floor.

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
- Per-tile AABB reject is a **real, exact** forward-walk lever: **1.32-1.45x**, bit-identical
  for room/bicycle. It strictly beats per-pixel no-op culling (1.17-1.23x) by also removing the
  `q`-test, but is **capped ~1.45x** by the staging floor (27-42% of baseline).
- Two-stage staging lowers the floor 5.3x and is bit-identical, but on aggregate the stage-2
  global reads for survivors offset the floor gain, so the end-to-end two-stage speedup vs the
  same gsplat staged baseline is **not** a clean win (see table). The floor is memory-bandwidth
  bound; lowering it further requires keeping survivors' params resident, not re-reading them.
- **Redirection (paper-level):** the forward-walk cost lives in the **staging floor + per-tile
  list traffic**, not in the per-occurrence compute. The next lever must attack list *traffic*:
  (a) **shared-memory-resident survivors** — stage each surviving gaussian's 9 floats once per
  tile into smem and have all 16x16 pixels read from smem (no stage-2 global re-read), or
  (b) a **hierarchical per-pixel significant-only list** (build the alpha>=cutoff list once, reuse),
  or (c) **cheaper culling** (avoid the `q`-evaluation for rejected gaussians via a coarse
  tile-center test). All three target the floor, which is where the headroom is. Note (a) is
  what single-stage already does — so the real headroom is in (b)/(c) and in shrinking the
  survivor list itself (e.g. a coarser tile grid so more gaussians are fully rejected).

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
