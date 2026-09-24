# External Research Queue — ChatGPT Priority Search

**Date:** 2026-10-19  
**Purpose:** Candidates requiring external prior-art verification before any implementation decision.  
**Context:** DSH cannot perform reliable Web/arXiv/GitHub search. These candidates are prepared for ChatGPT evaluation in the next research phase.

---

## ⚠️ Important Note

**All three candidates below require PRIOR-ART CHECK before implementation.** No candidate currently receives a "strong candidate" recommendation from local evidence alone. The queue is ordered by (a) potential impact and (b) how urgently the prior-art question needs to be resolved to unblock local decisions.

---

## Candidate #1 (Top Priority): C17-2 v2 — Two-Phase Sorting

### Core Mechanism

Break the single global CUB radix sort into two phases:
- **Phase A**: Narrow-key CUB sort by `(tile_id | image_id)` only (14-bit key, 4 passes instead of 12)
- **Phase B**: Per-tile depth sort (bitonic in shared memory or CUB segmented sort on tile segments)

The mechanics are fully documented in `v2_design_review.md`. Key challenge: Phase B cost uncertainty (occupancy collapse for large tiles) prevented the local gate from passing.

### Search Queries

```text
- Gaussian Splatting two-phase sort hierarchical tile-group
- 3DGS tile-then-depth segmented sorting
- differentiable Gaussian splatting local tile sort
- 3DGS per-tile block-level sort CUDA
- gsplat sorting optimization hierarchical sort
- 3DGS radix sort key reduction tile group first
- tile-local depth sort differentiable renderer
```

### Why Prior-Art Risk Matters

Phase 17A (`phase17a_candidate_design.md`) states:
> **NOT_EXISTING** in any differentiable 3DGS renderer. The standard approach (both gsplat and diff-gaussian-rasterization) uses a single global radix sort. The two-phase approach is a novel adaptation of hierarchical sorting ideas.

However, the Phase 17A author had no external search capability either. The claim "NOT_EXISTING" must be verified externally.

**If prior art exists**: We need to understand whether the existing approach handles the occupancy problem (Phase B cost) that our design review identified as the gate failure.

**If no prior art exists**: The candidate still faces the local GATE FAILURE — Phase B cost uncertainty was the reason for rejection, not lack of prior art.

### External Verification Needed

| Question | Why It Matters |
|:---------|:---------------|
| Does any published paper implement tile-group-first sorting for 3DGS? | Confirms/falsifies the "no prior art" claim |
| How do existing papers handle per-tile sort cost for large tiles (>2048 entries)? | Could provide known solutions to Phase B occupancy problem |
| Is there published data on CUB DeviceSegmentedRadixSort efficiency for small segments (100–5000 items)? | Could bound Phase B cost without CUDA implementation |
| Do any papers use block-level bitonic sort for 3DGS intersection data? | Direct prior art for Phase B mechanism |
| Are there differentiable renderers using per-primitive rather than global sorting? | Broader prior art that might apply to 3DGS |

### Potential Research Gap to Investigate

The Phase B occupancy problem (bitonic sort of 1024+ items in shared memory with ≤4 blocks/SM) is a fundamental GPU occupancy challenge. If no published work addresses this specific trade-off between CUB global radix sort efficiency and per-tile local sort efficiency for 3DGS, it represents a real research gap — but the gap may be "this approach doesn't work well on GPUs" rather than "nobody has tried it."

---

## Candidate #2 (Medium Priority): C1 — Depth Key Compression

### Core Mechanism

Truncate the depth field in the CUB sort key from full 32-bit float32 to the upper 16 bits only. This reduces CUB `end_bit` from 46→30 (16-bit reduction, ~33% fewer sort passes for RADIX_BITS=4).

The patch is complete: `patches/IntersectTile.c1.cu`. Source-verified: 5 lines changed. P5 CUDA verification is BLOCKED by toolchain incompatibility (PyTorch 2.7.1 + gsplat 1.5.3 + CUDA 13.3 + MSVC).

### Search Queries

```text
- RoofGS depth key compression 3DGS sort
- Gaussian Splatting depth quantization sort key
- 3DGS truncated depth sort IEEE 754 ordering
- "Taming 3DGS" integer depth encoding sort
- 3DGS radix sort key width reduction quantized depth
- depth bit reduction Gaussian splatting sorting
- float32 upper bits sort key 3DGS
```

### Why Prior-Art Risk Matters

The C1 gate review (`c1_gate_review.md` §1) claims:
> **C1's approach (truncate high 16 bits of depth in sort key) is novel in the 3DGS literature.**

This claim was made WITHOUT external search. The report differentiates C1 from RoofGS on specific technical grounds (no scene bounds, no division/clamp, fixed 16-bit upper-half truncation vs RoofGS's quantization formula), but this differentiation needs independent verification.

The prior-art question is what determines whether C1 is:
- **(A) A publishable contribution** if truly novel in 3DGS, or
- **(B) A known technique** that should be benchmarked against existing work

### External Verification Needed

| Question | Why It Matters |
|:---------|:---------------|
| Does RoofGS or any other paper measure sort-time speedup from depth compression? | The key C1 claim (33% fewer passes → measurable speedup) needs external evidence |
| Are there non-3DGS papers using float32 upper-bit truncation for GPU sort ordering? | Confirms general technique applicability |
| Does Speedy-Splat or TC-GS use any sort key compression? | These were excluded from Phase 17A literature scan |
| Is there published PSNR analysis of depth-collision effects on 3DGS rendering quality? | C1's ~5–15% ordering collision rate needs external validation of acceptable impact |

### Potential Research Gap to Investigate

If C1's approach (no scene bounds, no normalization, fixed 16-bit upper-half truncation) is NOT found in any 3DGS paper, AND no published work measures the sort-time speedup from depth key compression, this represents a clean, measurable research gap. The optimization is trivial to implement but its systematic evaluation as a sort acceleration technique appears undocumented.

---

## Candidate #3 (Lower Priority): Multi-Camera Segmented Sort

### Core Mechanism

Using gsplat's existing `segmented=True` parameter for multi-image (I>1) batched renders. The segmented sort sorts per-image segments independently, narrowing the key from `32 + tile_n_bits + image_n_bits` to `32 + tile_n_bits` (excludes image bits).

For single-camera (I=1), Phase 14B proved this is **1.9–4.5× slower** — the CUB segmented sort overhead exceeds the narrower-key benefit. The multi-camera case (I≥2) has never been tested.

### Search Queries

```text
- 3DGS multi-camera training segmented sort
- gsplat multi-view batch rendering optimizations
- Gaussian Splatting batched training multi-image sort
- CUB DeviceSegmentedRadixSort multi-segment efficiency
- multi-view 3DGS training sort optimization
- differentiable Gaussian splatting batch camera sort
```

### Why Prior-Art Risk Matters

Segmented sort is a gsplat built-in feature (`segmented=True`). It's NOT a new candidate — it's an under-evaluated one. The prior-art question is:
- Is multi-camera training with segmented sort documented anywhere?
- What speedup (if any) does the community observe on multi-view training?

### External Verification Needed

| Question | Why It Matters |
|:---------|:---------------|
| Is there published multi-camera training data for gsplat with segmented sort? | Directly answers whether this is worth investigating |
| How many cameras are typical in multi-view 3DGS training? | Determines segment count and potential benefit |
| Do any papers discuss CUB segmented sort performance for 3DGS specifically? | External performance characterization |

### Potential Research Gap

The single-camera finding (segmented sort is 1.9–4.5× slower) is documented in Phase 14B. The multi-camera case is a straightforward gap: nobody has published the results for I≥2.

---

## Summary

| Rank | Candidate | Urgency | Local Status | External Question |
|:----:|:----------|:-------:|:-------------|:------------------|
| **1** | C17-2 Two-Phase Sort | **HIGH** | GATE FAILED (Phase B cost) | Is there published work on per-tile depth sort for 3DGS that solves the occupancy problem? |
| **2** | C1 Depth Compression | **MEDIUM** | BUILD BLOCKED | Is upper-16-bit depth truncation in 3DGS sorting truly undocumented? What PSNR impact is acceptable? |
| **3** | Segmented Sort (I>1) | **LOW** | UNTESTED for multi-camera | What speedup on multi-view training? How many cameras are typical? |

---

*End of External Research Queue*
