# ClockGS — C0 Geometry-Clock Convergence Oracle: Final Report

**Lane**: ClockGS — Adaptive Multi-Rate Differentiation for Gaussian Splatting
**Verdict**: `DROP_MULTI_SCENE`
**Date**: 2026-10-04
**Env**: Windows, RTX 5070 Laptop 8GB shared, torch 2.13.0+cu130, CUDA 13.0, Python 3.13

---

## Verdict (preregistered decision rule)

C0 quality gate (preregistered): compare each k_geo variant against the k_geo=1
control on the SAME scene; require |dPSNR| <= 0.15 dB AND |dSSIM| <= 0.005.
"The lane only continues if at least k_geo >= 4 passes on room, garden, bicycle."
"If k_geo=4 fails on any scene: CLOCKGS = DROP_MULTI_SCENE. STOP."

Result: **k_geo=4 FAILS on bicycle** (dPSNR=-0.350 dB, dSSIM=-0.0181). Both also
fail at k_geo=2. => **DROP_MULTI_SCENE**. C1-C6 NOT executed (protocol STOP).

---

## OBSERVED

C0 is a training-dynamics oracle: the full backward is computed every iteration;
geometry (xyz/scaling/rotation) optimizer steps occur only once every k_geo iters
(geometry `.grad=None` on skipped iters => Adam freezes geometry state exactly, no
momentum drift). Opacity + SH step every iter. Densification runs to
`densify_until_iter`; the clock runs over `densify_until_iter -> end_iter`.

Authoritative results (append-only `clockgs/runs/c0/c0_log.jsonl`, aggregated by
`clockgs/src/_agg_c0.py`):

| scene  | schedule (densify->clock) | k_geo | PSNR    | dPSNR   | SSIM    | dSSIM   | gate |
|--------|---------------------------|-------|---------|---------|---------|---------|------|
| room   | 8K->12K                   | 1     | 32.3565 |  (ctrl) | 0.9571  |  (ctrl) |  -   |
| room   | 8K->12K                   | 2     | 32.4257 | +0.0692 | 0.9567  | -0.0004 | PASS |
| room   | 8K->12K                   | 4     | 32.2807 | -0.0758 | 0.9553  | -0.0018 | PASS |
| garden | 8K->12K                   | 1     | 29.0316 |  (ctrl) | 0.9145  |  (ctrl) |  -   |
| garden | 8K->12K                   | 2     | 29.0094 | -0.0222 | 0.9134  | -0.0011 | PASS |
| garden | 8K->12K                   | 4     | 29.0469 | +0.0153 | 0.9133  | -0.0012 | PASS |
| bicycle| 4K->8K (mem-bounded)      | 1     | 26.0504 |  (ctrl) | 0.8033  |  (ctrl) |  -   |
| bicycle| 4K->8K (mem-bounded)      | 2     | 25.8295 | -0.2209 | 0.7935  | -0.0098 | FAIL |
| bicycle| 4K->8K (mem-bounded)      | 4     | 25.7004 | -0.3500 | 0.7852  | -0.0181 | FAIL |

Per-test-image dispersion (n_test): room 39, garden 24, bicycle 25; psnr_std ~2.4-2.8,
ssim_std ~0.02-0.065. bicycle deltas exceed BOTH thresholds at k_geo=2 AND k_geo=4,
monotonically (1->2->4 PSNR 26.05->25.83->25.70; SSIM 0.803->0.794->0.785).

Clock mechanism verified: `drift.json` records 1001 geometry-grad records over the
4K->8K clock (every 4th iter), with non-zero xyz_norm (~0.06), scaling_norm (~0.0013),
rotation_norm (~0.004) => geometry IS stepped on schedule and frozen otherwise.

### Disambiguation (bicycle confound check)

bicycle was memory-bounded to an EARLIER clock window (4K->8K) because its
2.88M-gaussian model OOM-thrashes at 8K densification on 8GB; room/garden clocked
at 8K->12K (converged). To test whether bicycle's failure is a STAGE effect
(early clocking) or scene-INTRINSIC, room+garden were re-run at bicycle's exact
4K->8K schedule (separate log `clockgs/runs/c0_disamb/c0_disamb_log.jsonl`):

| scene | schedule | k_geo | PSNR | dPSNR | SSIM | dSSIM | gate |
|-------|----------|-------|------|-------|------|-------|------|
| room  | 4K->8K    | 1     | 31.8806 | (ctrl) | 0.9523 | (ctrl) | - |
| room  | 4K->8K    | 4     | 31.5617 | -0.3189 | 0.9486 | -0.0037 | FAIL(PSNR) |
| garden| 4K->8K    | 1     | 28.6735 | (ctrl) | 0.9064 | (ctrl) | - |
| garden| 4K->8K    | 4     | 28.3905 | -0.2830 | 0.8994 | -0.0070 | FAIL |

Room stage-effect: SAME scene PASS late (8K->12K, dPSNR=-0.076) but FAIL early
(4K->8K, dPSNR=-0.319). The geometry clock tolerance is strongly training-STAGE
dependent, not purely scene-intrinsic.

## EVIDENCE

- Authoritative log: `clockgs/runs/c0/c0_log.jsonl` (append-only; `run_c0.py`
  overwrites `c0_results.json` each launch, so the jsonl is authoritative).
- Aggregation: `clockgs/src/_agg_c0.py` -> `clockgs/runs/c0/c0_eval_all.json`.
- C0 harness: `clockgs/src/core.py` (`train_clock`), `clockgs/src/run_c0.py`,
  `clockgs/src/bootstrap.py` (wires canonical graphdeco 3DGS; shims simple_knn +
  fused_ssim).
- Per-run drift: `clockgs/runs/c0/{scene}_kgeo{1,2,4}/drift.json`.
- Disambiguation: `clockgs/src/run_c0_disamb.py`,
  `clockgs/runs/c0_disamb/c0_disamb_log.jsonl`.
- Canonical rasterizer: Inria `diff_gaussian_rasterization` prebuilt
  `_C.cp313-win_amd64.pyd` (4-tuple forward, no `antialiasing` field).
- Seed=0, images_8, identical camera sequence / optimizer / hyperparameters across
  k_geo within each scene (only the geometry clock period differs).

## INFERENCE

- Geometry (xyz/scale/rotation) tolerates 4x-slower updates on room and garden
  (small/medium scenes, converged 8K->12K clock window): deltas within gate.
- bicycle FAILS the gate at k_geo=2 and k_geo=4, monotonically. bicycle is the
  largest / most geometry-dense scene tested (1.87M gaussians at the 4K clock).
- The failure is CONFOUNDED by the memory-bounded schedule: bicycle clocked at
  4K->8K (geometry still rapidly coarsening) vs room/garden at 8K->12K (converged).
- DISAMBIGUATION (room at bicycle's exact 4K->8K schedule): room FAILS at 4K->8K
  (dPSNR=-0.319) but PASSED at 8K->12K (dPSNR=-0.076). => The failure is largely a
  STAGE effect: clocking geometry EARLY (while still coarsening) hurts far more
  than clocking it LATE (converged). bicycle is memory-bounded to the early window,
  so it fails; room/garden pass only because they were given the late window.
- Garden DISAMBIGUATION (4K->8K): FAILS (dPSNR=-0.283, dSSIM=-0.0070) but PASSED
  at 8K->12K (dPSNR=+0.015). => Stage effect GENERALIZES across room AND garden.

### Stage-effect summary (same scene, two clock windows)

| scene  | late window 8K->12K dPSNR | early window 4K->8K dPSNR | effect |
|--------|---------------------------|---------------------------|--------|
| room   | -0.0758 (PASS)            | -0.3189 (FAIL)            | strong stage effect |
| garden | +0.0153 (PASS)            | -0.2830 (FAIL)            | strong stage effect |
| bicycle| (cannot run 8K->12K: 2.88M gaussians OOM) | -0.3500 (FAIL) | memory-bounded to early window |

ROOT CAUSE (inferred): the geometry clock tolerance is training-STAGE dependent.
Clocking geometry EARLY (4K->8K, while xyz/scale/rotation are still rapidly
coarsening and the topology is still settling) breaks convergence; clocking it
LATE (8K->12K, geometry near-converged) is well tolerated. bicycle is
memory-bounded (8GB shared VRAM) to the early window, so it fails; room/garden
pass only because they were given the late window.

## NOT_MEASURED

- C1 (attribute-family oracle), C2 (backward cost decomposition), C3
  (attribute-specialized backward), C4 (correctness), C5 (adaptive controller),
  C6 (authoritative timing) — NOT executed; protocol STOPs at C0 (k_geo=4 fails
  bicycle).
- k_geo = 8, 16, infinity — not tested (k_geo=4 already fails bicycle; higher
  clocks only worse).
- Instrumented rasterizer `clockgs/third_party/dgr_clock` (per-kernel CUDA timing
  + C3 skip-mode flags) was BUILT and COMPILES (BUILD_EXIT=0, imports with
  clk_set_timing/clk_set_mode/clk_get_times) but was NOT executed, because C0
  failed and C2/C3 are gated behind C0. Its correctness (FULL-mode == canonical)
  was therefore NOT validated.

## CONCLUSION

**DROP_MULTI_SCENE** (preregistered verdict). The geometry clock does not
preserve quality at k_geo>=4 across all three preregistered scenes: bicycle
exceeds both thresholds (|dPSNR|=0.35 > 0.15; |dSSIM|=0.018 > 0.005) and degrades
monotonically with the clock period (k_geo=1->2->4: 26.05->25.83->25.70 PSNR).
Per the preregistered gate, the lane stops at C0; C1-C6 not executed.

**Scientific refinement (disambiguation):** the failure is substantially a
STAGE effect, not a fundamental lack of headroom. room and garden BOTH pass when
clocked late (8K->12K, geometry near-converged: dPSNR=-0.076 / +0.015) and BOTH
fail when clocked early (4K->8K, geometry still coarsening: dPSNR=-0.319 /
-0.283). bicycle is memory-bounded (8GB shared VRAM; 2.88M gaussians OOM at 8K
densification) to the early window, so it fails. INFERRED: a stage-aware ClockGS
(clock late, or adapt the clock to geometry-convergence state) could plausibly
rescue the method, but that is beyond the preregistered C0 scope and the
uniform-schedule multi-scene gate fails as written. This is a constructive
negative result: the geometry clock has real headroom, but it is stage-gated.

ClockGS is independent of RetireGS; no RetireGS retirement was used to rescue
this result.

Raw data, scripts, and hashes: see `clockgs/runs/c0/`,
`clockgs/runs/c0_disamb/`, `clockgs/CLOCKGS_C0_REPORT.md`.
