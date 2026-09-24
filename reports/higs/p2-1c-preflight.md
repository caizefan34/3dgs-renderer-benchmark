# P2-1C-P0 — Exact Forward-State-Reuse Backward Preflight

**Date**: 2026-09-22 · **Environment**: `mx` (ssh mx), A100-PCIE-40GB, `higs-13scene-env` torch 2.9.1+cu128 · **Type**: PREPARATION / ORACLE ONLY — no production kernel modified, no speedup claimed, no deployment authorized

**Decision: `P2_1C_PREFLIGHT_WEAK`**

---

## 1. Mission

Prepare the complete scientific basis for P2-1C: *reuse native HiGS forward hierarchy state to skip backward work that forward has already proven inactive*. P2-1C implementation remains conditional on **P2-1A correctness PASS + P2-1A runtime STRONG/MARGINAL**.

## 2. Method

Three inputs:

1. **Source audit** of the P2-1A native forward ([MacroTileRasterize.cu](file:///c:/Users/36570/3dgs-renderer-benchmark/remote_work/native_adapt/MacroTileRasterize.cu)) and the C0 V3 frozen backward ([HigsNativeBackward.cu](file:///c:/Users/36570/3dgs-renderer-benchmark/remote_work/native_adapt/HigsNativeBackward.cu) + SCALAR_ADJOINT / H8-MR patches).
2. **Exact measurement** on mx for room/bicycle/garden, cam0, 2048 (`scripts/p2/p2_1c_measure.py`; h5-0 capture methodology: authoritative B2 forward + CPU structural macro projection).
3. **Prior measured evidence** from H5-0R, which already *implemented* exact frontier load-gating (H5A/H5B/H5C) on the actual C0 V3 kernel with a 20w/100m/5r noise-envelope protocol.

## 3. Forward state usable (Results 1-2)

| State | Granularity | Where | Survives kernel? | P2-1C usable? |
|---|---|---|---|---|
| `smem_masks[mb*32+tile]` | (32-G mini-batch × fine tile) M0 coverage | MacroTileRasterize.cu L426 | No (shared mem) | Would need new global persistence |
| `out_active_mask[mt_batch_idx]` | (1024-G macro-batch × fine tile) M1 OR | L477 | Yes (global) | Too coarse (any single hit activates tile) |
| `last_ids` | per-pixel absolute flatten index | forward outputs | Yes | Already consumed by C0 V3 (`bin_final`) |
| `mt_gauss_offsets / ids / batch_offsets` | per-macro-tile sorted lists | global | Yes | Structure only |

**Semantic level: M0** — `GetMacroTileVisibilityMask` is a geometric fine-tile coverage test (`center_hit \|\| edge_hit`, `q <= t_rast`, MacroTileRasterize.cu L141-195). It is **not** alpha-accepted (M2) and **not** composited-before-termination (M3). No M2/M3 mask is materialized anywhere. Using M0 as the dead predicate is conservative and therefore exact.

## 4. Exactness proofs (Results 3-6)

| Stage | Can dead group be skipped? | Proof |
|---|---|---|
| **Blend backward** | **YES (exact)** | Dead(G) ⇒ no M0 coverage ⇒ forward never fetched the group for this tile ⇒ zero blend contribution ⇒ zero pixel-adjoint gradient. Or the group lies beyond every pixel's `bin_final` (BW L204-210), already skipped in processing. |
| **Projection VJP** | **YES (group-local, exact)** | Gradients are `atomicAdd` sums over (pixel, entry) terms (BW L270-290). Zero terms are removable; gaussians appearing in other tiles accumulate there. H8-MR moment form unchanged for active entries. |
| **SH VJP** | **YES (exact, independent)** | Consumes only `colors_eval`/`radii` (BW L994-1091); dead groups add zero `v_colors_eval`; no interaction with per-tile masks. |
| **Densification / means2d** | **SAFE** | No absgrad in this path (`compute_abs=false`, BW L256). `v_means2d` for active entries unchanged. H5-0R verified all 7 outputs within float noise (v_means2d ~1e-7; grad_means ≤1e-3 relative) for exact gating variants. |

## 5. Measured dead-group opportunity (Results 7-9, 12)

| Scene | Groups dead (matches P2-0 / H5-0) | **Dead entries (weighted)** | Dead 32-G by nocover / by lastid | Pure-suffix tiles | Internal dead/tile (mean) |
|---|---|---|---|---|---|
| room | 26783 / 127968 = **20.9%** | 102878 / 953144 = **10.8%** | 14994 / 11789 | **59.6%** | 1.27 |
| bicycle | 95213 / 316656 = **30.1%** | 66084 / 1412192 = **4.7%** | 86094 / 9119 | 35.8% | 7.65 |
| garden | 14182 / 71560 = **19.8%** | 43538 / 533928 = **8.2%** | 7362 / 6820 | **85.8%** | 0.34 |

**Key correction**: group counts massively overstate the opportunity. Most dead groups are *nocover* groups (zero entries — they cost the backward nothing), and groups are unequal in work. Weighted by entries, the forward-aligned 32-G dead work is **4.7-10.8%**, not 20-30%.

## 6. last_id overlap and incremental work (Results 10-11)

**Overlap = 100% on all scenes.** Every dead-group entry lies beyond the tile's max `last_id`, i.e. beyond every pixel's `bin_final` — so C0 V3 already skips **all** of the dead-group processing via `warp_bin_final`/`bin_final`. P2-1C's incremental value is strictly **load + control avoidance**.

The frozen C0 V3 kernel (`higs_blend_bwd_px_kernel`, PX=2, **block_size=128**) loads a batch's 128 gaussians (7 loads/entry, 40 B/entry) **unconditionally** (BW L433-451). A 128-G batch is load-skippable only if its *entire* index range lies beyond the frontier:

| Scene | Fully-dead 128-G batches | Entry fraction removable at frozen granularity |
|---|---|---|
| room | 30 / 12829 (0.23%) | **0.40%** |
| bicycle | 221 / 16659 (1.33%) | **2.00%** |
| garden | 45 / 10973 (0.41%) | **1.08%** |

## 7. Metadata design (Results 13-15)

- **M-A** per-(mini-batch,tile) dense bit: 55.0 / 235.3 / 43.8 KB per frame. Full generality (captures internal holes).
- **M-B** uint16 last-active-mini-batch frontier per tile: **22.0 / 22.0 / 21.2 KB per frame**. Exact (never skips an active group), captures trailing dead groups — which are 59.6% / 35.8% / 85.8% of tiles.
- **M-C** compact active-list: variable, equals M-A bound; more forward write work.

**Best: M-B** (scalar frontier). Metadata tax (write+read, ~44 KB/frame): **1.1-2.4% of 32-G savings** (passes ≤10%), but **3.9-28.6% of the frozen 128-G savings** (fails on room/garden). Note: none of this metadata exists globally in P2-1A today — `smem_masks` is transient.

## 8. Cost mapping and measured reality (Section 8)

`skip before load` vs `skip after load` is the correct distinction:

- **After load (processing)**: already 100% removed by C0 V3 `warp_bin_final`/`bin_final`.
- **Before load**: P2-1C would remove the unconditional cooperative smem load (BW L433-451) for fully-dead batches — at best 2.0% of entries at frozen granularity.

**H5-0R already implemented and timed the equivalent exact gates** (H5A whole-batch / H5B per-entry / H5C 32-G frontier) on the C0 V3 kernel: removed 1.6-13.8% of loads → **measured timing gain 0.00-1.54%, 0/3 scenes ≥1%** (room 1.54%, bicycle -0.01%, garden 0.00%). Root causes: processing already skipped; cooperative loads sit behind a `block.sync()` barrier (slowest thread dominates); removed loads are L1/L2-cached; the kernel is compute-bound.

## 9. Integration map (no patch) (Result 16-17)

- **Earliest safe gating point**: top of the per-batch loop in `higs_blend_bwd_px_kernel` (BW L430-432), *before* the first `block.sync()` of the load block. Predicate is **block-uniform** (`batch_end - 127 > L_tile_local`, `L_tile_local` = tile-wide max last_id) ⇒ zero divergence, barriers stay paired.
- **Forward write point** (future P2-1C): transition phase warp0 in MacroTileRasterize.cu L449-479 already holds per-tile `any_hit`; a uint16 max-active-mini-batch frontier per tile is a ~22 KB/frame new output.
- **ABI entry**: new const buffer pointer alongside `tile_offsets`/`flatten_ids`/`last_ids` in the backward args (BW L78-82 analog) and an `out` field in `MacroTileRasterizeArgs` (L198-212).
- **Scalar/H8 composition**: the gate is a block-uniform pre-check before the processing block; the SCALAR_ADJOINT recurrence, H8-MR moment accumulation and transmittance state are untouched for taken batches. Composes cleanly — does not replace them.

## 10. Corrected opportunity and gate (Results 18-20)

**P2-0's 13-20% gain does not survive.** It was based on group-count dead fractions. Corrected:

- forward-aligned 32-G design (requires new kernel structure): **4.7-10.8%** entry loads removable,
- frozen C0 V3 kernel: **0.4-2.0%**,
- measured timing for the equivalent exact gate: **0-1.5%** (H5-0R).

**Classification: `P2_1C_PREFLIGHT_WEAK`** — incremental dead work <8% on ≥2/3 scenes at best case (bicycle 4.7%, garden 8.2%), 0.4-2.0% at frozen granularity; metadata tax exceeds 10% of savings at frozen granularity; and the equivalent gate was measured at 0-1.5% timing.

**Implementation plan if P2-1A passes** (does not authorize deployment): add the uint16 frontier write in forward transition → thread the pointer through the backward ABI → add the block-uniform early continue at the batch-loop top → run the correctness oracle (all 8 tensors within the noise envelope) → run the timing protocol (20w/100m/5r balanced). Given measured 0-1.5% timing and <10% projected gain, **deployment is not warranted even if P2-1A passes**.

## 11. Deliverables

- Report: this file (`reports/higs/p2-1c-preflight.md`)
- Artifacts (`artifacts/higs-p2-1c-p0/`): `provenance.json`, `forward_state_semantics.json`, `exactness_proof.json`, `dead_group_counts.json`, `last_id_overlap.json`, `suffix_vs_holes.json`, `backward_work_mapping.json`, `metadata_candidates.json`, `metadata_cost.json`, `source_integration_map.json`, `correctness_oracle_spec.json`, `timing_protocol.json`, `final_gate.json` (+ supporting `weighted_dead_work.json`, `batch256_work.json`)
- Measurement script: `scripts/p2/p2_1c_measure.py` (ran on mx, run ids `9dd9fa7d-df5` / `*` on cuda:7, cuda:3)
