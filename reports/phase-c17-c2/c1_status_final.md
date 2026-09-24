# C1 — Final Status Report

**Status:** FROZEN / ARCHIVED-PENDING

**Date:** 2026-09-06

**Research classification:** Engineering adaptation / differentiated validation candidate (NOT novel sorting-key compression mechanism)

---

## 1. Evidence Chain Status

| Phase | Gate | Status | Notes |
|:------|:-----|:------:|:------|
| P1 | Prior art differentiation | ✅ COMPLETED | `c1_prior_art_differentiation.md` — verified C1 differs from RoofGS in depth encoding (no division, no scene bounds, fixed 16-bit truncation vs resolution-adaptive quantization) |
| P2 | Source audit | ✅ COMPLETED | `c1_source_audit_final.md` — baseline and C1 key construction, CUB end_bit, offset shifts, and downstream consumers verified from source. 16-bit end_bit reduction confirmed (46→30 bits global, 45→29 bits segmented). |
| P3 | Minimal Python prototype | ✅ COMPLETED | `c1_minimal_verification_gate.md` — Python smoke test with arXiv-quality PSNR/SSIM (room 34.6 dB / garden 39.6 dB / bicycle 36.9 dB). **Not formal CUDA evidence** — tile-level render, no pixel-center evaluation. |
| P4 | Implementation gate | ✅ COMPLETED | C1 patch written (5-line IntersectTile.cu change + offset + CUB end_bit). Design documented in `c1_comparative_design.md`. |
| P5 | CUDA sort performance | ❌ BLOCKED | `c1_p5_sort_performance.md` — accessible Linux/A100 (`mx`) could not build baseline gsplat CUDA backend reproducibly due to combined PyTorch/gsplat JIT API mismatch + CUDA 11.5/GCC 11 C++17 incompatibility. **NEGATIVE** for this validation attempt only. |
| P5.1 | CUDA build (baseline + C1) | ❌ BLOCKED | |
| P5.2 | Isolated CUB SortPairs timing | ❌ NOT EXECUTED | Prerequisite (P5.1) not met |
| P5.3 | Forward timing (baseline + C1) | ❌ NOT EXECUTED | |
| P5.4 | CUDA correctness (image comparison) | ❌ NOT EXECUTED | |
| P6–P9 | Backward / gradient / quality / training | **NOT REACHED** | P5 blocked prevents forward validation |

---

## 2. What C1 Actually Changes

C1 is a narrow 16-bit depth-key-width experiment in the IntersectTile CUDA kernel:

| Property | Baseline | C1 | Source-verified? |
|:---------|:---------|:---|:-----------------|
| Depth bits in key | 32 (full float32 bitcast) | 16 (upper bits of bitcast) | ✅ Lines 98–103 |
| Key composition | `iid_enc \| (tile_id << 32) \| depth_id_enc` | `depth_upper \| (tile_id << 16) \| iid_enc` | ✅ Lines 108 (base), 113 (C1) |
| CUB global end_bit | `32 + tile_n_bits + image_n_bits` | `16 + tile_n_bits + image_n_bits` | ✅ Lines 322 (base), 328 (C1) |
| CUB segmented end_bit | `32 + tile_n_bits` | `16 + tile_n_bits` | ✅ Lines 377 (base), 383 (C1) |
| Offset kernel shift | `>> 32` | `>> 16` | ✅ Lines 227 (base), 233 (C1) |
| **Total bits sorted (global)** | **46** (1080p, tile16) | **30** (1080p, tile16) | ✅ |
| **Bit reduction** | — | **16 bits** | ✅ |

The 16-bit reduction in CUB end_bit reduces the CUB radix sort key width by 16 bits. The exact pass-count reduction depends on CUB's runtime policy (bits-per-pass parameter) and is not determined by source alone.

---

## 3. What C1 Does NOT Do

- Does NOT change quantization formula (no division, no clamp, no z_near/z_far)
- Does NOT change tile_n_bits or image_n_bits computation
- Does NOT change the intersection pass itself (same intersect kernel, same counting)
- Does NOT change flatten_ids or tile_offsets semantics
- Does NOT change the rasterization kernel (forward or backward)
- Does NOT change tile size, tile count, or geometric intersection count
- Does NOT introduce a novel mechanism — it is an engineering adaptation of the sort key layout

---

## 4. What C1 Share with RoofGS

- **Depth-width reduction in sort key** — both reduce depth bits below 32
- **Both keep the upper portion of depth** (most significant bits dominate ordering)

## 5. What Differentiates C1 from RoofGS

| Property | C1 | RoofGS (2608.15785) |
|:---------|:---|:--------------------|
| Depth encoding | `u32(float32) >> 16` (no division) | `clamp(floor((d-z_n)/(z_f-z_n)*(2^b_d-1)), 0, 2^b_d-1)` |
| Depth bits | Fixed 16 | Variable: `32 - ceil(log2(N_tiles))` (19 @ 1080p) |
| Scene bounds needed? | No | Yes (z_near, z_far) |
| Tile bits | Fixed (same as gsplat) | Resolution-adaptive |
| Rasterizer changes? | No | Yes (fast-exp, dual-pixel) |
| Differentiable? | Yes (backward unchanged) | No (inference only) |
| Key width container | 64-bit (30 used) | 32-bit (fully used) |

---

## 6. Runtime Hypothesis (Unvalidated)

C1 hypothesizes that reducing the CUB radix sort key width from 32 to 16 depth bits **reduces sort time proportionally to the per-pass bit-width reduction for whatever CUB bits-per-pass policy is used at runtime**. This is a pure sort-work reduction hypothesis: fewer bits sorted per element → fewer passes over the data → less global memory traffic → lower sort wall time.

**This hypothesis is NOT falsified by the available evidence.** It is also NOT confirmed. The runtime impact of the reduced sort key width remains unvalidated because the target CUDA backend could not be built reproducibly.

**Should NOT be stated as:** "Reduced sort key width did not translate into meaningful runtime improvement"

**Should be stated as:** "Reduced sort key width was not validated for runtime effect on the accessible target because a reproducible CUDA baseline could not be established."

---

## 7. Re-entry Condition

Re-open this C1 investigation only if **all** of these conditions are met:

1. A Linux/A100 environment with a known-compatible gsplat/PyTorch/CUDA toolkit/compiler combination (e.g., gsplat 1.5.3 + PyTorch 2.4 + CUDA 12.1 + GCC 9/10/12, or gsplat built from source against matching toolchain)
2. A reproducible unmodified gsplat CUDA JIT build and baseline forward pass
3. The C1 patch can be applied, rebuilt, and compared without environment debugging

**Do not** invest further effort in:
- Repairing the current `mx` environment's toolchain mismatch
- Windows/MSVC JIT compatibility
- Installing alternative CUDA toolkits or GCC versions without sudo
- Docker/container setup on `mx` (no images found)

---

## 8. Documents

| Document | Status |
|:---------|:-------|
| `reports/phase-c17-c2/c1_prior_art_differentiation.md` | ✅ Final |
| `reports/phase-c17-c2/c1_source_audit_final.md` | ✅ Final |
| `reports/phase-c17-c2/c1_minimal_verification_gate.md` | ✅ Final (Python-only smoke test) |
| `reports/phase-c17-c2/c1_comparative_design.md` | ✅ Final (design, Python fallback, old blockers) |
| `reports/phase-c17-c2/c1_p5_sort_performance.md` | ✅ Updated (NEGATIVE/INCONCLUSIVE) |
| `reports/phase-c17-c2/c1_status_final.md` | ✅ THIS FILE |
| `patches/IntersectTile.c1.cu` | ✅ Patch preserved (contains stale pass-count comments — do not use as evidence) |
| `tools/c1_minimal_verification.py` | ✅ Python smoke test tool preserved |
| `tools/p5_baseline_runner.py` | ✅ Baseline timing runner preserved (requires working CUDA backend) |
| `tools/p5_c1_runner.py` | ✅ C1 timing runner preserved (requires working CUDA backend) |
