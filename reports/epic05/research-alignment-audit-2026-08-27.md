# EPIC-05 Research Alignment Audit

**Date:** 2026-08-27  
**Phase:** 6 (Research Realignment)  
**Audit Scope:** Full repository — all experiments, reports, source code, and results  
**Mode:** Analysis only — no code modifications

---

## Executive Summary

This audit realigns the EPIC-05 research program from a performance-driven tile-size study to a **differentiability-gated optimization research platform** as required by the advisor.

**Current state:** The repository has excellent performance evidence for tile_size effects across A100 and RTX 5070 (supporting 5 hypotheses), quality Layer-A pixel equivalence (confirmed), and partial Layer-B GT quality. However, it has a **critical evidence gap**: **zero gradient correctness verification across all optimization modules**.

**Key finding in one sentence:**  
> **`tile_size=32` has never been tested for gradient correctness, and no module in the repository has passed `torch.autograd.gradcheck` or finite-difference validation — this is the single highest-priority gap.**

---

## Q1: Current Repository — Independent Optimization Modules

Based on real code audit, not abstraction.

### Module Registry (Real, Independently Switched)

| ID | Module | Mechanism | Independent Switch | Switch Location | Status |
|----|--------|-----------|:-----------------:|-----------------|--------|
| **M0** | Baseline (gsplat default) | Full 3DGS pipeline | N/A (reference) | `tile_size=16, packed=True, SH degree=3, eps2d=0.1` | Reference |
| **M1** | tile_size | Runtime launch parameter, changes CUDA grid dimensions and per-block threads | Yes | `gsplat.rasterization(tile_size=N)` — same kernel binary | **Confirmed: same binary, runtime param** |
| **M2** | packed/dense | Skip invisible Gaussians per camera (packed) vs all-Gaussian (dense) | Yes | `packed=True/False` parameter | Existing |
| **M3** | SH degree | Number of spherical harmonic coefficients used for color | Yes | `sh_degree=0/1/3` parameter | Existing |
| **M4** | radius_clip | Near/far clipping of projected Gaussian screen-space radius | Yes | `radius_clip` parameter | Existing |
| **M5** | eps2d | Additive blur to 2D covariance for numerical stability | Yes | `eps2d` parameter | Existing |
| **M6** | HiGS tile_size | HiGS inference renderer with tile size variants | Yes | `GaussianInferenceRenderer(tile_size=N)` | Separate renderer class |
| **M7** | HiGS SH compression | FP16 compression of SH coefficients (32b, 16b) | Yes | `sh_compression="none"/"32b"/"16b"` | Separate renderer class |
| **M8** | HiGS auto adapter | Scale-aware heuristic: <300K → tile16+SHnone, >=300K → tile8+SH32b | Yes (derived) | `GsplatHiGSAutoRenderer` | Adapter |
| **M9** | TC-GS | Tensor Core matrix alpha computation (separate backend) | Yes | Separate CUDA extension | External repo |

### Modules NOT Independently Switched (Not Optimization Modules)

| Item | Reason |
|------|--------|
| Kernel fusion | No independent switch — baked into gsplat implementation |
| Memory reuse | Buffer reuse is in renderer wrapper, not independently ablated |
| Block size (M6 in protocol.md) | Not a real module — no `block_size` parameter in current gsplat API |
| Densification/pruning | Part of training — not a renderer optimization module |
| Projection optimization | No independent switch — part of gsplat core |
| Sorting optimization | No independent switch — part of gsplat core |

### Edge Cases: Boundary Modules

- **HiGS** is a separate renderer, not an optimization *of* gsplat — but tile_size and SH compression within HiGS are independent modules.
- **TC-GS** is a separate repository altogether — included as a benchmark candidate, not as a controlled optimization module.

---

## Q2–Q4: Forward, Backward, and Gradient Correctness

### Q2: Which Modules Have Forward Correctness Evidence?

| Module | Forward Correctness Evidence | Best Available Evidence |
|--------|:---------------------------:|------------------------|
| M1 tile16 vs tile32 | **SUPPORTED** | Bit-exact on bicycle/garden; max pixel error 0.002 on room (floating-point reduction order) |
| M1 tile8 vs tile16 | **SUPPORTED** | Bit-exact on bicycle/garden |
| M2 packed vs dense | **INCONCLUSIVE** | Performance ±0.5%; no pixel-level equivalence published |
| M3 SH degree (0/1/3) | **PARTIALLY_SUPPORTED** | Performance measured; quality not formally compared |
| M4 radius_clip | **INCONCLUSIVE** | Performance ±0.1%; no quality impact study |
| M5 eps2d | **INCONCLUSIVE** | Performance ±0.1%; no quality impact study |
| M6 HiGS tile_size | **PARTIALLY_SUPPORTED** | Pixel equivalence vs gsplat dense has PSNR check (~59 dB) but formal equivalence not established |
| M7 HiGS SH compression | **PARTIALLY_SUPPORTED** | SH32: 64.85 dB min; SH16: 49.79 dB min — quality drop known |
| M9 TC-GS | **PARTIALLY_SUPPORTED** | Within quality thresholds vs original 3DGS in smoke test |

### Q3: Which Modules Have Backward Correctness Evidence?

| Module | Backward Correctness | Evidence |
|--------|:-------------------:|----------|
| ALL M1–M9 | **NOT_TESTED** | No formal backward correctness verification exists for any module |

**Note:** The gsplat library itself has backward kernels, so backward *executes* (no crash), but numerical correctness of backward *under different module configurations* has never been verified.

### Q4: Which Modules Are Gradient-Correct?

| Module | Gradient Correctness | Evidence |
|--------|:-------------------:|----------|
| ALL M1–M9 | **NOT_TESTED** | **Zero gradient correctness evidence in the entire repository** |

**Critical finding:** There is no `torch.autograd.gradcheck` call, no finite-difference comparison, and no gradient numerical analysis anywhere in:
- `tests/` (all 11 test files)
- `scripts/epic05/` (all experiment scripts)
- `src/` (analysis, benchmark, renderer code)

---

## Q5: Ground-Truth Quality Evidence

### Layer A — Renderer Correctness (candidate vs baseline)

| Comparison | Max Pixel Error | Mean Pixel Error | Grade |
|-----------|:--------------:|:----------------:|:-----:|
| tile16 vs tile32 (bicycle) | 0.0 (bit-exact) | 0.0 | **SUPPORTED** |
| tile16 vs tile32 (garden) | 0.0 (bit-exact) | 0.0 | **SUPPORTED** |
| tile16 vs tile32 (room) | 0.002481 | ~1e-5 | **SUPPORTED** (floating-point reduction order artifact) |
| tile8 vs tile16 (bicycle) | 0.0 (bit-exact) | 0.0 | **SUPPORTED** |
| tile8 vs tile16 (garden) | 0.0 (bit-exact) | 0.0 | **SUPPORTED** |

### Layer B — Reconstruction Quality (candidate vs GT)

| Scene | tile16 PSNR | tile32 PSNR | tile16 SSIM | tile32 SSIM | GT Provenance |
|-------|:----------:|:----------:|:----------:|:----------:|:-------------:|
| bicycle | — | — | — | — | Pending — requires dataset images |
| garden | — | — | — | — | Pending — requires dataset images |
| room | — | — | — | — | Pending — requires dataset images |

**Critical finding:** The Phase 3 official validation report (`official-dataset-validation-2026-07-19.md`) is a **template with pending cells** — the quality evaluation Python pipeline exists but has NOT been executed. The Phase 3 report (`epic05-phase3-official-validation-2026-08-18.md`) reports pixel equivalence but NOT GT quality numbers.

**Status: Layer B quality is BLOCKED/INCONCLUSIVE for all scenes.**

*Note: There is a `validate_quality.py` quality evaluation pipeline that does compute PSNR/SSIM/LPIPS against GT images, but it has not been run specifically for tile_size comparison on official scenes.*

### Training Quality

| Module | Training PSNR | Training SSIM | Training LPIPS | Quality Gate |
|--------|:-----------:|:------------:|:-------------:|:-----------:|
| M1 tile16 (room, 5000 steps) | 4.77 dB | — | — | **Random GT — not meaningful** |
| M1 tile32 (room, 5000 steps) | 4.77 dB | — | — | **Random GT — not meaningful** |

**Critical finding:** All training validation uses **random ground truth images** (`create_random_gt()`), making PSNR values meaningless for absolute quality assessment. The identical PSNR between tile16 and tile32 only proves that both converge to the same quality w.r.t. random target — not that they produce good reconstructions.

---

## Q6–Q10: Training Validation

### Training Hierarchy Classification

| Experiment | Classification | Evidence | Includes |
|-----------|:-------------:|----------|----------|
| Phase 2 A100 `--stage training` | **SANITY_CHECK** | 500 steps, single synthetic scene, MSE loss only | Forward, backward, Adam optimizer |
| Phase 4 RTX 5070 room (1000 steps) | **SIMPLIFIED_TRAINING** | Real scene, 1000 steps, MSE loss, no densification | Forward, backward, Adam optimizer |
| Phase 5 RTX 5070 room (5000 steps) | **SIMPLIFIED_TRAINING** | Real scene, 5000 steps, MSE loss, no densification | Forward, backward, Adam, stage analysis |
| Phase 5 RTX 5070 garden (5000 steps) | **SIMPLIFIED_TRAINING** | Real scene, 5000 steps | Forward, backward, Adam |

**Training components actually present:**
- ✅ Forward
- ✅ Backward (loss.backward)
- ✅ Adam optimizer
- ✅ PSNR tracking
- ❌ **Densification**
- ❌ **Pruning**
- ❌ **Adaptive Gaussian control**
- ❌ **SH degree progression**
- ❌ **Real GT images (uses random targets)**
- ❌ **Full training loss (L1 + D-SSIM)**
- ❌ **Validation split**

### Full Training Evidence Matrix

| Optimization | Forward | Backward | Densification | Pruning | Optimizer | Full Step | Quality |
|-------------|:-------:|:--------:|:-------------:|:-------:|:---------:|:---------:|:-------:|
| M1 tile16 baseline | ✅ | ✅ | ❌ | ❌ | ✅ | ✅* | ❌ |
| M1 tile32 | ✅ | ✅ | ❌ | ❌ | ✅ | ✅* | ❌ |
| M2 packed/dense | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| M3 SH degree | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| M4–M5 rclip/eps2d | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| M6–M8 HiGS | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

*\* "Full step" = forward + backward + optimizer; does not include densification/pruning.*

### Inference vs Training tile Preference

| GPU | Scene | Inference Best | Training Best | Consistent? |
|:---:|:-----:|:--------------:|:-------------:|:-----------:|
| A100 | 50K synthetic | tile32 | tile32 | ✅ Yes |
| A100 | 400K synthetic | tile32 | Not tested | N/A |
| RTX 5070 | room | tile16 | tile32 (median) | ❌ **No** |
| RTX 5070 | garden | tile16/ambivalent | Not completed | N/A |

**Phase 5 finding:** On RTX 5070 room, inference and training optimal tiles **differ** — inference favors tile16 while training favors tile32 in median timing. This contradicts earlier assumptions that inference preference generalizes to training.

---

## Q7–Q8: Composability

### Current Interaction Experiments

The 10 interaction experiments (I1–I10) tested combinations of:
- tile8/16/32 + packed/dense
- tile8/16 + SH0
- dense + SH0
- tile8 + packed + SH0
- tile8 + rclip

**Result: No meaningful interactions detected.** All non-tile modules have <0.5% individual effects, so their combinations also show negligible effects.

### Composability Gate Status

| Combination | Forward | Backward | Gradient | Quality | Pass Gate? |
|------------|:-------:|:--------:|:--------:|:-------:|:----------:|
| tile32 + dense | INCONCLUSIVE | NOT_TESTED | NOT_TESTED | SUPPORTED | ❌ |
| tile32 + SH0 | INCONCLUSIVE | NOT_TESTED | NOT_TESTED | PARTIAL | ❌ |
| tile32 + rclip | INCONCLUSIVE | NOT_TESTED | NOT_TESTED | INCONCLUSIVE | ❌ |

**No combination passes the composability gate because gradient correctness has never been tested for any module individually, let alone in combination.**

---

## Q9: Optimization Value Classification

### Inference-Only Value (Faster inference but untested gradient)

| Module | Inference Speedup | Gradient Tested? | Classification |
|--------|:----------------:|:----------------:|:--------------:|
| M1 tile32 (A100) | 1.42×–3.93× | **NOT_TESTED** | **Inference finding only** |
| M1 tile16 (RTX 5070) | Baseline | Baseline | Reference |
| M6 HiGS tile8 | Significant vs gsplat | **NOT_TESTED** (inference-only renderer) | **Inference finding only** |
| M7 HiGS SH32 | Modest (P99 improvement) | **NOT_TESTED** | **Inference finding only** |
| M9 TC-GS | ~1.33× vs gsplat dense | **NOT_TESTED** | **Inference finding only** |

### Training-Validated Value (Has training evidence)

| Module | Training Speedup | Training Quality | Full Pipeline? | Classification |
|--------|:--------------:|:----------------:|:--------------:|:--------------|
| M1 tile32 (A100, synthetic 50K) | 1.22× | Identical PSNR | ❌ Simplified | **PARTIAL training evidence** |
| M1 tile32 (RTX 5070, room) | 0.93× (median) | Identical PSNR | ❌ Simplified | **PARTIAL training evidence** |

---

## Q10: Current Strongest Candidate Modules for Contribution

Ranked by alignment with advisor requirements:

### Tier 1: Highest Potential (Strong Performance + Partial Pipeline)

1. **M1 tile_size=32 (A100)** — 1.42×–3.93× inference, 1.22× simplified training speedup, pixel-identical output. **But: gradient correctness is UNKNOWN.** If gradient check passes, this becomes the strongest candidate.

### Tier 2: Moderate Potential (Needs More Validation)

2. **M1 tile_size (hardware-aware)** — The finding that inference preference is GPU-resource-dependent is a valid architectural contribution, but training preference may differ from inference. Needs gradient + full training validation.

### Tier 3: Low Contribution Potential (Small Effect or Blocked)

3. **M2 packed/dense** — ±0.5% effect — not a meaningful contribution
4. **M3 SH degree** — ±0.3% effect — not a meaningful contribution
5. **M4 radius_clip** — ±0.1% — not a meaningful contribution
6. **M5 eps2d** — ±0.1% — not a meaningful contribution
7. **M7 HiGS SH compression** — SH32 minimal quality loss, SH16 significant loss
8. **M9 TC-GS** — External repo, not a controlled module

---

## Q11: Largest Evidence Gaps (Priority Order)

### P0: Correctness / Gradient Correctness

| Gap | Severity | Current Status |
|-----|:--------:|:--------------|
| **G0: Tile_size gradient correctness** | **CRITICAL** | NOT_TESTED |
| **G1: All modules gradient correctness** | **CRITICAL** | NOT_TESTED for all 9 modules |
| **G2: Finite-difference gradient verification** | **CRITICAL** | NOT_TESTED for any parameter |
| **G3: Autograd gradcheck** | **CRITICAL** | NOT_TESTED anywhere in repo |

### P1: Full Training Validation

| Gap | Severity | Current Status |
|-----|:--------:|:--------------|
| G4: Full training with densification | **HIGH** | NOT_TESTED (all training is simplified) |
| G5: Training with real GT (not random) | **HIGH** | NOT_TESTED |
| G6: Full training loss (L1 + D-SSIM) | **HIGH** | NOT_TESTED |
| G7: Training across multiple scenes | **HIGH** | Only room + partial garden |
| G8: Training quality (PSNR/SSIM/LPIPS vs GT) | **HIGH** | NOT_TESTED |

### P2: Ground-Truth Quality

| Gap | Severity | Current Status |
|-----|:--------:|:--------------|
| G9: Layer B GT quality for tile16 (all scenes) | **MEDIUM** | BLOCKED — requires dataset images |
| G10: Layer B GT quality for tile32 (all scenes) | **MEDIUM** | BLOCKED — requires dataset images |
| G11: GT quality gates documentation | **MEDIUM** | Thresholds defined but not applied |

### P3: Composability

| Gap | Severity | Current Status |
|-----|:--------:|:--------------|
| G12: tile32 + packed gradient | **MEDIUM** | NOT_TESTED (prerequisite: G0) |
| G13: tile32 + SH degree interaction gradient | **MEDIUM** | NOT_TESTED (prerequisite: G0) |

### P4: Performance Optimization

| Gap | Severity | Current Status |
|-----|:--------:|:--------------|
| G14: Nsight Compute stall analysis | **LOW** | BLOCKED (WDDM permission) |
| G15: Third GPU cohort | **LOW** | Not planned |
| G16: Larger workloads (1M+ Gaussians) | **LOW** | Not needed for current contribution |

---

## Q12: Next Experiments

### Decision: `READY FOR TARGETED EXPERIMENT`

**After the audit, the decision is to proceed with targeted experiments — but only for the highest-priority gaps. All further blind optimization is halted.**

### Top 1: Gradient Correctness for tile_size

**Experiment name:** `epic06-gradcheck-tile-size-v1`  
**Priority:** P0  
**Why:** This is the single largest gap. If tile_size=32 produces incorrect gradients, it cannot be a final candidate. If gradients are correct, it immediately becomes the strongest contribution.

**Protocol:**
1. Use `torch.autograd.gradcheck` on `gsplat.rasterization()` with tile_size=16 and tile_size=32
2. For each parameter group (means, scales, rotations, opacity, SH):
   - Analytical gradient from backward pass
   - Finite-difference gradient (central, eps=1e-4, 1e-5, 1e-6)
   - Report: `max_abs_error`, `max_relative_error`, `mean_relative_error`
3. Run at 1080p with synthetic scene (50K Gaussians) — single camera
4. Test both packed and dense modes
5. Record: parameter_group, dtype, seed, camera_count

**Expected output:**
```
reports/epic05/gradient-correctness-YYYY-MM-DD.md
results/epic05/gradient/gradcheck_tile_size.json
```

**Success criteria:** `max_relative_error < 1e-4` for all parameter groups at both tile_size=16 and tile_size=32.

### Top 2: Ground-Truth Quality for tile_size

**Experiment name:** `epic06-gt-quality-tile-size-v1`  
**Priority:** P2 (but codependent with Top 1 for completeness)  
**Why:** Layer B quality (renderer vs GT) is currently all BLOCKED. Without GT quality numbers, no claim of "quality preservation" can be made.

**Protocol:**
1. Run `validate_quality.py` for tile16 and tile32 on all 3 official scenes
2. Use official Mip-NeRF 360 GT images
3. Report: PSNR, SSIM, LPIPS per scene per tile size
4. Compare: tile16 vs GT, tile32 vs GT

**Note:** Top 1 must complete first — if gradients are incorrect, quality evaluation is moot.

### Top 3: Simplified Training with Real GT

**Experiment name:** `epic06-simplified-training-real-gt-v1`  
**Priority:** P1  
**Why:** Current training uses random GT. This must be corrected to real GT before any training claims are meaningful.

**Protocol:**
1. Run current simplified training (Phase 5 script) with **real GT images** instead of `create_random_gt()`
2. Compare tile16 vs tile32 on room scene at 5000 steps
3. Report: step time, forward/backward breakdown, PSNR trajectory (now meaningful)
4. No densification/pruning (same simplified training, just real GT)

---

## Summary: Optimization-to-Contribution Pipeline

```
Module identified
    ↓
Forward correctness [PASS: tile_size]
    ↓
Backward correctness [PASS: gsplat backward exists, but tile32 variants UNKNOWN]
    ↓
Gradient correctness [P0 — NOT TESTED — must be first experiment]
    ↓
Quality preservation (Layer A: pixel equivalence) [PASS: tile_size]
Quality preservation (Layer B: against GT) [P2 — BLOCKED — requires images]
    ↓
Simplified training with real GT [P1 — NOT DONE — random GT used currently]
    ↓
Full training with densification/pruning [P1 — NOT DONE]
    ↓
Composability verification [P3 — depends on gradient correctness]
    ↓
End-to-end training benefit claim [Final goal]
```

**Every module above M0 is currently stuck between "backward correctness" and "gradient correctness" in this pipeline.**

---

## Final Verdict

```
╔══════════════════════════════════════════════════════════╗
║             RESEARCH ALIGNMENT AUDIT RESULT              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  DO NOT OPTIMIZE YET                                     ║
║                                                          ║
║  Reason: Gradient correctness is the gate. Without it,   ║
║  no optimization module qualifies as a "differentiable   ║
║  3DGS training optimization."                            ║
║                                                          ║
║  Current strongest candidate: tile_size                  ║
║    - Performance: STRONG (1.42×–3.93× inference)         ║
║    - Forward correctness: SUPPORTED (pixel-identical)    ║
║    - Quality Layer A: SUPPORTED (bit-exact or near)      ║
║    - Backward correctness: NOT_TESTED (gsplat backward   ║
║      exists, but tile32-specific gradient not checked)   ║
║    - Gradient correctness: NOT_TESTED ★★★                ║
║    - Quality Layer B: BLOCKED (no GT execution)          ║
║    - Full training: NOT_TESTED                           ║
║                                                          ║
║  Readiness: READY FOR TARGETED EXPERIMENT                ║
║    - 1 experiment only: gradient correctness for         ║
║      tile_size (Top 1)                                   ║
║    - If gradient correct, run Top 2 and Top 3            ║
║    - If gradient incorrect, tile_size is invalidated      ║
║      as a training contribution (inference-only finding) ║
║                                                          ║
║  What to avoid:                                          ║
║    ❌ More tile heuristics                               ║
║    ❌ More interaction experiments                       ║
║    ❌ New optimization invention                         ║
║    ❌ Performance optimization of any kind               ║
║    ❌ Nsight Compute stall analysis (not a gap)          ║
║                                                          ║
║  What to do:                                             ║
║    ✅ Gradient correctness (gradcheck + finite diff)     ║
║    ✅ If gradient correct: GT quality + training         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

### Performance Findings (valid but not training contributions)

The following are **confirmed performance findings** but not **candidate research contributions** until gradient correctness is verified:

1. `tile_size=32` on A100: 1.42×–3.93× inference speedup (synthetic scenes, 50K–400K)
2. `tile_size=16` on RTX 5070: best inference on official scenes
3. Hardware-aware tile preference: A100 prefers tile32, RTX 5070 prefers tile16
4. Same kernel binary: tile_size is a runtime parameter (cuobjdump evidence)
5. Thread capacity, not shared memory, explains the tile_size preference (Phase 5 finding)
6. Inference preference ≠ training preference (RTX 5070 room: tile16 inference, tile32 training)

### Candidate Research Contributions (requires gradient correctness)

Only after gradient correctness is verified:

1. **tile_size as a correct, differentiable optimization** for A100 training (if gradient check passes)
2. **Hardware-aware tile selection heuristic** grounded in SM thread capacity
3. **Inference vs training tile preference divergence** finding

### Invalidated Candidates

| Module | Reason for Invalidation |
|--------|------------------------|
| M2 packed/dense | ±0.5% effect — too small to matter |
| M3 SH degree | ±0.3% effect — too small to matter |
| M4 radius_clip | ±0.1% effect — too small to matter |
| M5 eps2d | ±0.1% effect — too small to matter |
| M7 HiGS SH16 | Quality drop to 49.79 dB — unacceptable |

---

## Appendices

### A. File Audit Summary

| Directory | Files | Relevance |
|-----------|:-----:|-----------|
| `benchmark/` | — | Not present (not a top-level directory) |
| `benchmark_suite/` | Suite configs | Dataset manifests |
| `src/` | 40+ Python files | Benchmark infrastructure, renderer adapters |
| `src/renderers/` | 9 files | Renderer adapters including gsplat, HiGS, TC-GS |
| `tests/` | 11 files | No gradient tests |
| `scripts/epic05/` | 30+ scripts | All experiment scripts — no gradient check |
| `configs/epic05/` | 2 JSON | Optimization matrix, validation matrix |
| `results/epic05/` | Subdirectories | Raw, aggregated, final validation, phase4, phase5, official |
| `reports/epic05/` | 6 reports | All performance/quality focused — no gradient report |
| `docs/` | 19 docs | Architecture, taxonomy, protocol — no gradient protocol |
| `paper/` | — | Not yet present |

### B. Key Source Files Checked for Gradient Testing

```python
# File                          # Gradient tests present?
src/renderers/base.py           # None
src/renderers/gsplat_renderer.py # None
src/benchmark_framework/        # None
tests/test_renderers.py         # None
tests/test_quality.py           # None
tests/test_evaluation.py        # None
scripts/epic05/*.py             # None (all 30+ scripts)
```

### C. gsplat Native Backward Kernels (from cuobjdump)

| Kernel | REG | SHARED | Purpose |
|--------|:---:|:-----:|---------|
| `rasterize_to_pixels_3dgs_bwd_kernel<float, SH=3>` | 48 | 1,024 B | Main backward of rasterization |
| `spherical_harmonics_bwd_kernel<float>` | 40 | 0 B | SH gradient |
| `projection_ewa_3dgs_packed_bwd_kernel<float>` | 95 | 0 B | Projection backward |

These exist and execute during `loss.backward()`, but their **numerical correctness under tile_size=32** has never been verified against:
- tile_size=16 output
- Analytical finite-difference calculation

### D. Quality Pipeline Status

- `src/scripts/validate_quality.py`: ✅ Exists, supports PSNR/SSIM/LPIPS
- Official dataset GT images: ❌ Not confirmed on disk at `data/official/mipnerf360/*/images/`
- Quality evaluation executed: ❌ Not done for tile_size comparison
- Quality gates documented: ✅ Thresholds defined (`max_psnr_drop=0.1`, `max_ssim_drop=0.001`, `max_lpips_increase=0.001`)

---

*End of audit. This document should be read alongside `results/epic05/research_alignment_matrix.json` for machine-readable evidence status.*
