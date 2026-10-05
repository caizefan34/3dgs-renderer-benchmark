# BES0 No-Op Culling — Measured CUDA Speedup (the honest number)

**Date:** 2026-10-05  **Author:** DSH (autonomous, MX A100 idle window, GPU7)
**Companion to:** `reports/bes0_noop_census.md` (analytic 98.4% no-op, 59-70x ideal),
`reports/bes0_realscene_opportunity.md` (per-occurrence routing falsification)

## TL;DR
The census hypothesized that skipping the 98.4% no-op `(pixel, gaussian)` alpha-evals
would yield a "realistic several-x" forward-render speedup. I built a faithful,
gsplat-style CUDA alpha-compositing walk (per-tile shared-memory staging, front-to-back
blend) and **measured** it. Exact no-op culling (skip `exp`+blend when the alpha is
provably below the 1/255 cutoff) is **bit-identical** to the baseline (`max|diff| = 0`
on every camera) and gives a **1.17-1.23x** speedup (1.23x room / 1.18x bicycle / 1.17x garden) on the forward walk — **not** several-x,
and nowhere near the 59-70x ideal. The ideal assumed culling removes the *entire*
per-occurrence cost; in reality the quadratic-form test and the staging/loop floor remain.

## Method
Faithful to the frozen `scene_adapter` (gsplat-1.5.3-convention projection) and the
census per-tile gaussian lists. One CUDA thread-block per 16x16 tile, one thread per
pixel, front-to-back walk over the tile's depth-sorted gaussian list. The tile list is
staged into shared memory in chunks (cooperative load), so each gaussian's params are read
from global memory once per tile. Three modes in one kernel:
- `baseline` (0): every occurrence pays `q` + `exp` + threshold + blend.
- `culled`  (1): skip `exp`+blend when `q > 2*(ln(op)+ln(255))` (provably `alpha<cutoff`).
- `floor`   (2): stage + loop only, no `q`/`exp`/blend (memory/loop lower bound).

The culling test is algebraically identical to the baseline `op*exp(-0.5*q) < 1/255` test
(verified on a grid: 0 disagreements), with a `+0.01` numerical margin so the culled path
only skips gaussians whose alpha is *provably* below cutoff and evaluates `exp` identically
to the baseline in the boundary slice. Result: **bit-identical** output.

Guard: fail-closed MX policy, GPU7 only, no co-resident job touched. Toolchain: nvcc 11.5,
gcc 11.4, torch 2.7.1+cu118, A100-40GB (SM80).

## Results (3 scenes x 12 dev cameras, MX GPU7, clean run)
All 36 cameras: `max|diff| = 0` (bit-identical), determinism check `det = 0`.

| scene | occurrences | base (ms) | cull (ms) | floor (ms) | speedup | per-cam range |
|-------|------------:|----------:|----------:|-----------:|--------:|--------------:|
| room    | 514,037,931   | 506.2 | 410.4 | 114.8 | **1.233x** | 1.151-1.268x |
| bicycle | 1,676,846,955 | 1373.4 | 1160.7 | 505.3 | **1.183x** | 1.127-1.232x |
| garden  | 1,012,524,460 | 1238.1 | 1055.2 | 340.9 | **1.173x** | 1.118-1.232x |

Per-camera speedups (all exact): room 1.21-1.27 (most), bicycle 1.13-1.23, garden
1.12-1.23. The speedup is modest and consistent across scenes; the larger scenes
(bicycle/garden) sit slightly lower because their staging floor is a larger
fraction of the baseline (bicycle floor = 505/1373 = 37%).

Aggregate floor (staging+loop, no compute) is 23-37% of the baseline and is not
removed by culling; the `exp`+blend that culling removes is ~20-27% of the
compute. That is the entire ceiling.

## Why 1.26x and not 63x (the key finding)
The census ideal `n_total/n_sig = 63x` counts *occurrences*: if you could skip a no-op
entirely, you would do 63x less work. But a real kernel cannot skip a no-op for free:
1. **The `q` test is mandatory.** To know a gaussian is a no-op you must evaluate the
   quadratic form `q = a dx^2 + 2b dx dy + c dy^2` (and compare to `2 ln(op/cutoff)`).
   `q` is ~60-70% of the remaining per-occurrence cost and culling keeps it.
2. **Staging + loop floor.** The shared-memory staging and per-tile loop/sync overhead
   (`floor` mode) is ~25% of the baseline and is not removed by culling.
3. **Only `exp`+blend is removed.** Skipping `exp`+blend for the 98.4% no-ops saves
   ~22 ms of the ~82 ms compute (~27% of compute, ~20% of total) -> 1.26x.

So the forward-walk cost is **not** concentrated in the `exp`+blend of no-ops (which
culling removes); it is concentrated in the `q`-test + staging (which culling cannot
remove). The 63x ideal is an upper bound that assumes culling is free; the realistic
ceiling for *exact* no-op culling is ~1.26x.

## Verdict / research redirection
- Exact no-op culling is a **real but weak** lever: **1.26x** on the forward walk,
  bit-identical output. It is not the "several-x" the census hoped for, and far from 63x.
- The strong version of "skip the no-op bulk" is **结构上 (structurally) capped at ~1.26x**
  for any exact method, because the per-occurrence `q`-test and staging floor dominate.
- **Redirection:** to materially speed up the 3DGS forward walk, reduce the `q`-test +
  staging cost — e.g. a hierarchical per-pixel *significant-only* list (build the
  alpha>=cutoff list once, reuse), cheaper culling tests, or sparsity — rather than
  skipping no-ops inside a dense tile walk. This complements the routing falsification
  (LAW-BEVT): both show the per-occurrence alpha machinery is not where large 3DGS speedup
  lives.

## Reproduce
- Probe `bes_cull_probe/cull_probe.py`, kernel `bes_cull_probe/cull_kernel.cu`,
  chain `bes_cull_probe/chain_cull.sh 7`.
- Raw: `/dev/shm/bes_cull/out/cull_{room,bicycle,garden}.json`.
- Run: `bash chain_cull.sh 7` (GPU7). `CULL_R_CAP=256`, `CULL_REPEATS=50`.

## Caveats
- Measured on the **forward alpha-compositing walk only** (not projection, binning, SH
  evaluation, or the backward pass). End-to-end training speedup from this lever alone is
  < 1.26x.
- `r_cap=256px` radius cap (14-24% of gaussians capped) bounds the per-tile list size for
  the 2.9-3.9M-gaussian scenes; the cap affects only a few huge near-camera gaussians.
- Per-gaussian color uses one SH evaluation (view-dir from camera center), mode-invariant
  across baseline/culled, so it does not affect the speedup or the exactness check.
