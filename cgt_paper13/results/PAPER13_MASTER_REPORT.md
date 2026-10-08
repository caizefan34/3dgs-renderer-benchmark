# CGT-PAPER13 — Master Report: 13-Scene Renderer Paper Benchmark

**Lane:** CGT-PAPER13 (renderer paper-evidence; does NOT reopen G5/TI0/native training)
**Date:** 2026-10-04
**Host:** MX `bms-39468022-001`, 8x A100-PCIE-40GB, runtime `liaoyuanjun_mx0qual` (torch 2.7.1+cu118, gsplat 1.5.3, CUDA 11.8.89)
**Method:** frozen CGT-F0 (R=4, rigorous interval certificate GATE=0.02, kernel-correctness gate 2e-5, TILE=16, single fused CUDA kernel, seed 20261004)
**Final classification:** `DROP_RENDERER_GENERALIZATION`

---

## Core scientific question
> Does the already-proven fast CGT renderer generalize from the original 3-scene kernel evidence to
> the standard 13-scene 3DGS rendering benchmark, while preserving rendering quality and producing a
> meaningful full-frame speedup against a strong canonical baseline after all production costs are
> honestly accounted for?

**Answer: NO.** The mixed-forward kernel speedup is real but does not survive the full production
frame path.

---

## OBSERVED (raw measurements)
- **P0 Source audit:** 13/13 scenes have valid COLMAP data + canonical checkpoints; F0 ingests all 13
  schemas (`xyz/scaling/rotation/opacity/shs/active_sh_degree`, all sh_degree=3). No training required.
  Cohorts: 3 EXACT_COHORT_F0 (bicycle/garden/room) + 10 COMPATIBLE_CANONICAL_ABSGRAD (same trainer
  `gaussian_model_hash=b8cb0e25…`, git `32ab80e`, seed 42, ABSGRAD schedule).
- **P1 Canonical baseline (gsplat 1.5.3, packed, native res):** Baseline-B (per-scene fastest tile)
  geomean = **5.893 ms/frame (169.7 fps)**; global canonical tile = 32 (geomean 6.112 ms).
- **P2 CGT mixed-forward (50 tiles/scene, work-decile stratified, F0-style):**
  - S_mixed geomean = **1.2606x**, median 1.0216x, min 0.9706x (counter/truck), max 3.0319x (room).
  - Dataset geomean: mipnerf360 1.394x (n=9), tanksandtemples 0.985x (n=2), deepblending 1.028x (n=2).
  - Kernel correctness PASSES on all 13 scenes (max_abs 1.85e-6 << 2e-5 gate) at R=4.
  - Certificate: safe fraction mean 0.617, weighted fallback mean 0.451, rigorous pass rate mean 0.618.
  - Room reproduces F0: S_mixed 2.9457 vs F0 3.0087; view_idx and safe fraction (0.720) match exactly.
- **P3 E2E build-amortization (full-frame, per-camera views=1, extrapolated from 50-tile sample):**
  - Full-frame non-empty tiles: 4,108 (playroom) – 65,936 (treehill).
  - Per-frame build (fit R4 + certify) geomean = **40.3 min/frame** (range 0.9 min counter – 301.7 min treehill).
  - Full-frame CGT render is **6.70x SLOWER** than gsplat geomean (only counter 0.6x faster).
  - Novel-view E2E speedup S = **0.000 for all 13 scenes**; N_break = None for 12 scenes, 30,236 for counter.
- **P4 Ablations:** Frozen fused kernel is **R=4-specific** — R=8 (max_abs 0.864) and R=16 (max_abs 0.867)
  fail kernel correctness; the rank ablation cannot be validly run with the frozen kernel.

## EVIDENCE (traceable artifacts)
- `PAPER13_PREREG.json`, `PAPER13_METHOD_FREEZE.json` (hash-verified F0 config).
- `PAPER13_SCENE_REGISTRY.json`, `PAPER13_SOURCE_AUDIT.json` (per-scene SHA256, N, schema, cohort).
- `PAPER13_BASELINE_RESULTS.json` (P1), `PAPER13_CGT_RESULTS.json` (P2),
  `PAPER13_BUILD_AMORTIZATION.json` (P3), `PAPER13_CORRECTNESS.json`,
  `PAPER13_CERTIFICATE_STATS.json`, `PAPER13_ABLATIONS.json`, `PAPER13_VERDICT.json`.
- Raw per-scene: `results/p1/canonical_*.json`, `results/p2/cgt_*.json`, `results/p3/e2e_*.json`,
  `results/p2/raw_timing/` (raw per-launch samples), `results/_schema_probe.json`.
- Environment/GPU evidence in each per-scene JSON (gsplat/torch/device).


- Paper tables (section 33), source-traceable CSV: figures/TABLE1_13SCENE_MAIN.csv
  (per-scene baseline/S_mixed/safe/fallback/kernel-correctness/build/render-ratio/E2E),
  figures/TABLE2_DATASET_AGGREGATE.csv (mipnerf360 / tanksandtemples / deepblending geomean aggregates),
  figures/TABLE3_ABLATIONS.csv (rank R4/R8/R16 + cert/fallback),
  figures/TABLE4_MEMORY.csv (peak VRAM = NOT_MEASURED), figures/TABLE5_RELATED_WORK.csv
  (canonical gsplat vs CGT-F0 vs closed CGT-G5).
- Figure source data (section 32): figures/fig1_method_overview.csv (pipeline stages),
  figures/fig2_speedup_distribution.csv (per-scene S_mixed),
  figures/fig3_safe_fallback_vs_speedup.csv (safe/fallback fraction vs speedup),
  figures/fig4_rank_ablation.csv (R4/R8/R16 speed and kernel correctness).
- Environment/GPU provenance (section 38): captured in every per-scene P1/P2/P3 JSON
  (device id, torch 2.7.1+cu118, gsplat 1.5.3, CUDA 11.8.89) + host line above.

## INFERENCE (interpretation)
- The CGT student + rigorous certificate compress the per-tile render function, and the fused kernel
  IS faster than the teacher kernel on small work-decile samples (mixed-forward 1.26x geomean).
- The representation is **screen-space tile-based**: it must be rebuilt (fit + certify) per camera per
  frame. For novel-view synthesis (new camera every frame) this build is unavoidable and dominates (~40 min/frame).
- The tile-based fused render (one block/tile, no cross-tile sort) is ~6.7x slower than gsplat's
  optimized full-frame rasterizer, so even the render step loses to the strong canonical baseline.
- Therefore the kernel-level speedup does not survive the full production frame path.

## HYPOTHESIS (untested here)
- A camera-independent / view-amortized CGT representation could rescue E2E — but that is a NEW method,
  forbidden in this frozen lane.
- Higher-rank kernels (R=8/16) might change the speed/quality trade-off — blocked by the R=4-only kernel.

## NOT_MEASURED
- `PAPER13_MEMORY.json`: per-scene peak VRAM (NOT_MEASURED).
- Full-image GT dPSNR/dSSIM/LPIPS quality gates (NOT_REACHED; mixed-forward tile-level only).
- Ablation C (fused vs unfused) — NOT_MEASURED (no unfused reference kernel).
- Nsight hardware counters (NOT_MEASURED; no profiler permission assumed).
- Same-hardware external renderer reproduction (optional, section 29) — NOT_ATTEMPTED.

## CONCLUSION
`DROP_RENDERER_GENERALIZATION`. The frozen CGT-F0 renderer delivers a genuine, correct, certifiable
mixed-forward kernel speedup on small tile samples (geomean 1.26x, up to 3.03x on room), but this
benefit does **not** generalize to full-frame end-to-end rendering: the per-camera per-tile
fit+certificate build costs ~40 min/frame and the tile-based render is ~6.7x slower than canonical gsplat,
yielding novel-view E2E speedup ~0 across all 13 scenes. The result is consistent with the upstream
TI0 `DROP_UPDATE_COST` finding. CGT should be treated as a rendering representation with a real but
non-generalizing kernel speedup, not as a training primitive or a full-frame renderer accelerator.
Do not promote to a larger multi-scene TI1.