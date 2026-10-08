> **Protocol-grade BES-GS execution (this session):** see eports/bes0_bes_gs_execution.md — the frozen BES0 spec was executed faithfully (E0 PASS, E1 full-scene opportunity screen on 957.8M occurrences, E2/E3 NOT_RUN per gate-4). Verdict NO_GO_OPPORTUNITY (rho_R=1.04e-9, production gsplat pipeline confirmed). This ceiling document is the supporting broader analysis.

# BES0 — Paper-Level Ceiling Characterization: Bounded-Event Rendering Cost Reduction for 3DGS Training

**Date:** 2026-10-05  **Author:** DSH (autonomous continuation, MX A100, shared cluster)
**Companion evidence:** `reports/bes0_noop_census.md`, `reports/bes0_noop_culling_measured.md`,
`reports/bes0_realscene_opportunity.md`, `reports/bes0_headroom_curve.md`,
`reports/bes0_aabb_reject_measured.md`, `bes_aabb_ts32/aabb_{room,bicycle,garden}.json`,
git `65a0b41..4b84e83`, knowledge record `D:\学习\BES0_MX_DSH执行记录_20261005.md` 续12–续19.

## TL;DR
The BES0 target — **≥3× training-speedup at same quality via bounded-event rendering** — is
**not achievable** by rendering-side culling on real 3DGS. Across every exact and lossy lever
measured, the forward alpha-compositing walk tops out at **1.85× (bit-identical)** and the
end-to-end forward step is **1.30× measured** (a lower bound on gsplat-representative e2e; see dilution section). The cost is dominated by the per-occurrence
`q`-test (~60–70%) and the shared-memory staging floor (~23–37%), **not** by the `exp`+blend
(~20–27%) that exact culling can remove. BES0's core mechanism (bounded-event routing) is
**structurally empty** on real 3DGS (per-occurrence routing rate rho_R ≈ 1e-9). The protocol
verdict is therefore `INCONCLUSIVE_CUDA_NOT_IMPLEMENTED` (no production CUDA BES renderer).

## The BES0 ≥3× question
BES0 (bounded-event sampling + log-hazard jump + conditional integration + three-query training
path) hypothesizes that a 3DGS forward render evaluates many "events" (gaussian-pixel occurrences)
that provably cannot affect the output, and that skipping them yields a multi-x training speedup.
This report characterizes the *achievable* ceiling of that hypothesis family (rendering-side
occurrence culling) on the 3 frozen scenes (room 838k / bicycle / garden, 2.9–3.9M gaussians).

## Measured levers (forward alpha-compositing walk)
| Lever | Type | Speedup | Bit-identical | Source |
|---|---|---:|:---:|---|
| Per-occurrence bounded alpha routing (BES core) | exact | **0** (rho_R≈1e-9) | n/a | realscene_opportunity |
| Per-pixel transmittance early-stop | exact | 1.01–1.02 | yes | headroom_curve |
| No-op culling (per-pixel skip) | exact | 1.17–1.23 | yes | noop_culling_measured |
| Per-tile AABB reject, TS=16 | exact | 1.32–1.45 | 35/36 | aabb_reject_measured |
| **Per-tile AABB reject, TS=32** | exact | **1.63–1.85 (agg 1.742×)** | 35/36 @1% inflate; 36/36 @10% | aabb_ts32 |
| Tile + per-pixel reject (mode 4), TS=32 | exact | 1.72 | **no** (1.9e-3) | aabb_ts32 (this session) |
| Two-stage staging | exact | floor 5.3× lower; e2e not a clean win | yes | aabb_reject_measured |
| Thread coarsening (2×2 micro-tile) | exact | **negative** | - | aabb_reject_measured |
| Opacity cull B1A (mild, lossy) | lossy | 1.104 @ +0.110 dB | no (quality) | Campaign B |
| WARP_ALL (aggressive, lossy) | lossy | 0.37 @ −1.44 dB | no | Campaign B |

Aggregate AABB reject TS=32 (room, 12 dev cams): baseline 976.0 ms → tile-reject 527.2 ms = **1.851×**,
`max|diff| = 0` (bit-identical). The finer per-pixel reject (mode 4) is **worse** (1.723×) and
**introduces error** (1.9e-3) — the per-pixel `|dx|>dx_max` test adds overhead without enough extra
culling and is not a tight bound. So the tile-level AABB reject is the optimal exact lever.

## Mechanism: where the forward-walk cost actually lives
Decomposing the per-tile walk (cull_probe `floor` mode = stage+loop only, no compute):
- **q-test** (`q = a·dx² + 2b·dx·dy + c·dy²` vs `2·ln(op/cutoff)`): ~60–70% of remaining per-occurrence cost.
- **staging + loop floor** (cooperative shared-memory load, per-tile sync): ~23–37% of baseline.
- **exp + threshold + blend**: ~20–27% of baseline — the *only* part exact no-op culling removes.

Consequences:
1. No-op culling caps at ~1.26× because it removes only exp+blend; the mandatory q-test and staging floor remain.
2. AABB reject is stronger (1.85×) because it removes the **entire** per-occurrence cost (q-test + staging + exp + blend) for gaussians whose screen-space AABB misses the tile.
3. The cost is **memory-traffic bound** (staging floor), not compute-bound. This is why two-stage staging lowers the floor 5.3× but is not a clean end-to-end win (survivor global re-reads offset the gain), and why coarsening (a compute-shape change) is negative.

## End-to-end dilution (MEASURED this session)
All prior numbers were the **forward walk only**. A real training step is
`projection + binning + SH-eval + forward-walk + backward + densification`. We measured the
dilution directly with `bes_aabb_ts32/aabb_ts_probe_e2e.py` (TS=32, 50 repeats, A100,
bit-identical `max|diff|=0`), which times `setup = project_gaussians_2d + SH-eval + reshape +
AABB + tile-list build`, then reports `e2e_speedup = (setup + baseline) / (setup + tile)`
and `walk_fraction = baseline / (setup + baseline)`:

| scene | walk (speedup_tile) | setup_ms | baseline_ms | walk_fraction | e2e_speedup |
|---|---:|---:|---:|---:|---:|
| room | 1.846x | 655.8 | 674.9 | 0.507 | **1.303x** |
| bicycle | 1.855x | 1620.0 | 1714.9 | 0.514 | **1.311x** |
| garden | 1.635x | 1236.0 | 1717.9 | 0.582 | **1.292x** |

The measured end-to-end forward speedup is **~1.30x** on all three scenes. This is a
**lower bound**: the probe uses gsplat-convention *torch* projection (`sa.project_gaussians_2d`),
which is slower than gsplat's fused CUDA projection. A faster (smaller) setup would *raise*
e2e_speedup toward the 1.85x walk ceiling (as setup -> 0, e2e -> walk), so the gsplat-representative
e2e lies in **[1.30x, 1.85x]** and is still far below 3x. Garden is the weakest (walk 1.64x,
e2e 1.29x) because its thin overlapping foliage has a 22.5% never-skip fraction (AABB cannot
reject them). Data: `bes_aabb_ts32/e2e/aabb_{room,bicycle,garden}.json`,
`bes_aabb_ts32/e2e/e2e_{room,bicycle,garden}.log`.

The backward pass (VJP) and densification are unaddressed by forward culling, so full
training-step speedup is lower still. The realistic training number is therefore
**[~1.1x, 1.85x]**, with the measured forward-only lower bound at **1.30x**.

## Why 3× is unreachable via rendering-side bounded-event culling
- **Exact ceiling ~1.85× walk / 1.30× measured forward e2e (lower bound; gsplat-rep. ≤1.85×).** The dominant cost (q-test + staging) is not removable
  by exact culling; finer granularity (mode 4) is worse; two-stage/coarsening do not help end-to-end.
- **BES0 routing is structurally empty.** Real 3DGS gaussians are anisotropic sub-pixel thin sheets
  (room cam1: sigma_short ≈ 0.32 px, sigma_long ≈ 115 px). Any ≥2 px tile sees alpha underflow to 0
  at the tile far corners, so a bounded "event" (tile) contains ~0 rare/survivable gaussians:
  rho_R ≈ 1e-9 (ts=16), 0 (ts≤8). There is nothing to route.
- **Lossy does not reach 3× without wrecking quality.** Mild opacity cull (B1A) = 1.10× at +0.11 dB;
  aggressive (WARP_ALL) = 0.37× at −1.44 dB (slower *and* worse). The quality-speed curve is flat
  near 1× because the cost is in the alpha machinery of the *significant* gaussians, not a cullable
  no-op bulk.

## What would be required to reach 3×
1. **Reduce the q-test + staging cost** (the real bottleneck): cheaper culling tests, parameter
   sparsity, or a hierarchical significant-only list — but mode 4 shows naive finer granularity loses,
   so any hierarchy must beat the tile-level AABB bound *and* stay bit-identical.
2. **The actual BES backward-path method** (log-hazard jump + conditional integration) as a CUDA
   kernel — but per-occurrence routing was falsified on real 3DGS, so the mechanism has no substrate.
3. **Training-loop optimizations** (adaptive densification cadence, fewer iterations, resolution
   scheduling) — orthogonal to rendering, not "bounded-event sampling".

## BES0 protocol verdict (unchanged)
`INCONCLUSIVE_CUDA_NOT_IMPLEMENTED`: production CUDA BES renderer not implemented
(`cuda_implemented=false`); Torch-reference timings are not CUDA evidence. The paper-level finding
above characterizes *why* the ≥3× target is out of reach for this lever family even if the CUDA
renderer were built.

## Reproduce
- AABB reject TS=32: `bes_aabb_ts32/aabb_ts_probe.py` + `aabb_kernel_ts.cu`
  (`AABB_TILE_SIZE=32`, `CUDA_VISIBLE_DEVICES=<idle allowed GPU>`, guard-guarded). Modes 0/3/4.
- End-to-end dilution: `bes_aabb_ts32/aabb_ts_probe_e2e.py` (times setup=proj+SH+AABB+build, then e2e_speedup=(setup+baseline)/(setup+tile)). `bes_aabb_ts32/e2e/aabb_{room,bicycle,garden}.json`.
- CHUNK/staging-utilization test (ongoing): `bes_aabb_ts32/aabb_kernel_ts_c1024.cu`
  (`#define CHUNK 1024` + `cudaFuncSetAttribute` smem opt-in) vs `CHUNK 512`, via
  `/dev/shm/bes_chunk/chunk_runner.sh` — resolves whether the TS=32 per-occurrence penalty is
  staging half-utilization (report) or the serial front-to-back blend chain (续18).
- No-op culling: `bes_cull_probe/{cull_probe.py,cull_kernel.cu,chain_cull.sh}`.

## Caveats
- Walk-only + forward-e2e measured; backward (VJP) + densification unmeasured, so full training-step speedup is lower than the 1.30× forward lower bound.
- `r_cap=256px` radius cap (bounds per-tile list for the 2.9–3.9M-gaussian scenes; affects a few huge
  near-camera gaussians). Garden cam-68 needs 10% AABB inflation for bit-exactness (1% inflate is
  exact on 35/36).
- Shared MX cluster; timing uses min-based CUDA-event aggregates; speedup *ratio* is robust
  (median/min within 1–2%). Co-resident jobs were not touched.

## CHUNK / staging-utilization test (resolves the per-occurrence penalty mechanism)
The TS=32 per-occurrence penalty was diagnosed two contradictory ways: the AABB report attributed it
to **CHUNK=512 staging half-utilization** (1024 threads/block, CHUNK=512 → half the block idle during
cooperative staging); knowledge record 续18 attributed it to the **serial front-to-back blend chain**.
We tested this directly by rebuilding the kernel with `#define CHUNK 1024` (full staging utilization,
+ `cudaFuncSetAttribute` for the 53 KB dynamic smem), keeping TS=32, and timing mode 3 (tile reject)
against the CHUNK=512 baseline on 3 fresh idle A100 GPUs, all bit-identical (`max|diff|=0`):

| scene | CHUNK=512 (speedup) | CHUNK=1024 (speedup) | Δ |
|---|---:|---:|---:|
| room | 1.846x | 1.814x | -0.032 |
| bicycle | 1.849x | 1.807x | -0.042 |
| garden | 1.637x | 1.615x | -0.022 |

Full staging utilization gives **no speedup** — it is marginally *worse* (the 2x smem footprint adds
pressure). **Resolution:** the TS=32 per-occurrence penalty is the **serial blend dependency chain**
(`T *= (1-alpha)` is a serial prefix product; a 32px tile's list is ~4x longer than 16px → ~4x longer
per-pixel chain → expf pipeline stalls), **not** staging utilization. CHUNK=512 remains optimal; the
exact ceiling is unchanged at ~1.85x. This confirms 续18 and corrects the AABB report's staging claim.
(Data: `bes_aabb_ts32/chunk_test/aabb_{c512,c1024}_{room,bicycle,garden}.json`,
`aabb_kernel_ts_c1024.cu`; MX `/dev/shm/bes_chunk/`. Root `/` was 100% full, redirecting
`TORCH_EXTENSIONS_DIR`→storage-pool and `TMPDIR`→`/dev/shm` to build.)
