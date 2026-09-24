# C1 — Prior-Art Differentiation Audit

> **Purpose:** Determine whether C1 is a replication of existing work or a differentiated research/engineering question.
>
> **Scope:** RoofGS, TileGS, SplatShop, Faster-GS, and other 2025–2026 3DGS works with direct sorting / key compression / depth quantization overlap.
>
> **Date:** 2026-09-05

---

## 1. Methodology Comparison Table

**Constraint:** Only entries with source- or paper-verified data are included. Cells marked "—" indicate no evidence available.

| Aspect | gsplat Baseline | C1 (this proposal) | RoofGS (arXiv 2608.15785) | TileGS (arXiv 2609.03613) |
|--------|----------------|-------------------|---------------------------|---------------------------|
| **Sort key width** | 64-bit (46 bits used @ 1080p tile16) | 64-bit (30 bits used) | **32-bit** (full word) | 64-bit (same as baseline) |
| **Depth quantization** | None (full float32) | IEEE 754 **bitcast >> 16** (upper 16 bits of float32) | **Uniform linear** `floor((d-z_n)/(z_f-z_n)*(2^b_d-1))` (Eq 9) | None (full float32, same as baseline) |
| **Tile index bits** | `floor(log2(n_tiles))+1` = 13 (@1080p tile16) | Same (13) | `ceil(log2(N_tiles))` = 13 (@1080p tile16) | Same as baseline |
| **Depth bits** | 32 (full float32) | **16** (fixed: upper half of float32 bitcast) | **Variable:** 32 − tile_bits; 19 @1080p tile16 | 32 |
| **Image bits** | 1 | 1 | N/A (single-image pipeline) | 1 |
| **Key layout** | `img \| tile \| depth` (full 32b depth) | `depth_hi (16b) \| tile (13b) \| img (1b)` | `tile (MSB) \| quantized_depth (LSB)` | Same as baseline |
| **CUB sort end_bit** | Global: 46; Segmented: 45 | Global: **30**; Segmented: **29** | Paper states: **32** (full word) | Same as baseline |
| **CUB sort type** | `SortPairs` / `SegmentedSortPairs` | Same (patch only changes key) | Not specified (separate codebase) | Same as baseline |
| **Backward pass** | Yes (differentiable) | Yes (5-line key-only patch; downstream unchanged) | **Not addressed** (inference-only paper) | Yes (gsplat-based, No-GW variant) |
| **Training validated** | Yes | No (analysis only) | No | No |
| **Rasterization changed** | None | None (only sort key + offset shift) | **Extensive** (SH INT8, fast-exp, dual-pixel ILP, kernel fusion) | **Moderate** (tile-local binning, repair kernel) |
| **Reported speedup** | — | Not measured | **10.1× end-to-end** at 4K (RTX 4090) | 1.07–1.09× end-to-end |
| **Reported PSNR loss** | — | Not measured | **0.028 dB** (at 4K) | < 0.001 dB |
| **Scene bounds needed** | No | **No** | **Yes** (z_near, z_far) | No |

---

## 2. RoofGS — Detailed Analysis

RoofGS is the **strongest prior art** and the most directly relevant to C1.

### 2.1 Core mechanism

**Paper:** Luo et al., arXiv 2608.15785, Section IV-A2 "Resolution-Adaptive Sorting-Key Encoding"

> *"RoofGS compresses the sorting key into a 32-bit word that jointly encodes the tile index and a quantized camera-space depth."*

**Verified from paper text (Section IV-A2):**

- **Baseline used for comparison:** "64-bit composite key consisting of a 32-bit tile ID and a 32-bit floating-point depth" — this indicates RoofGS compares against **diff-gaussian-rasterization** (original 3DGS), not gsplat.
- **Baseline pass count (paper):** "This representation requires eight radix passes" — corresponding to 64-bit key sorted at 8 bits-per-pass.
- **RoofGS pass count (paper):** "cuts N_pass from 8 to 4" — corresponding to 32-bit key.
- **Key width:** 32 bits (full word)
- **Key layout:** Tile index (MSB) + quantized depth (LSB)

### 2.2 RoofGS quantization formula (Eq. 9, Section IV-A2)

```
q_d = clamp(floor((d - z_near) / (z_far - z_near) * (2^b_d - 1)), 0, 2^b_d - 1)
```

where:
- `z_near`, `z_far` = predefined scene extent (from camera frustum)
- `b_d` = `32 − b_t` (depth bits = total bits − tile bits)
- `b_t` = `ceil(log2(N_tiles))` (tile bits)

**Key properties:**
- Monotonically non-decreasing: "quantization cannot reverse the order of Gaussians assigned to distinct depth bins"
- Collision bound: "Ordering ambiguity is limited to Gaussians mapped to the same bin, whose depth difference is smaller than the quantization step Δ_d = (z_far − z_near)/(2^b_d − 1)"
- Resolution-dependent bit allocation (more tiles = fewer depth bits)

### 2.3 Overlap analysis with C1

| Dimension | C1 | RoofGS | Overlap type |
|-----------|----|--------|-------------|
| **Insight: compress sort key** | ✅ | ✅ | **Exact overlap** — both identify radix key size as bottleneck |
| **Quantization method** | IEEE 754 `>> 16` (logarithmic, fixed 16 bits) | Uniform linear `floor((d-z)/(range)*(2^b-1))` (19b @1080p) | **Partial overlap** — different math, different bits |
| **Scene bounds** | Not needed | Requires z_near/z_far | **Different** |
| **Backward/differentiable** | Supported (5-line gsplat patch) | Not addressed (inference-only framework) | **Different** |
| **Isolated sort speedup** | Yes (only key changed) | No (bundled with fusion, SH INT8, fast-exp, dual-pixel) | **Different** |
| **Baseline codebase** | gsplat (46-bit key) | diff-gaussian-rasterization (64-bit key) | **Different** |

### 2.4 What RoofGS validates (that C1 has not)

- ✅ End-to-end PSNR on real scenes (0.028 dB loss at 4K)
- ✅ Speedup across multiple GPUs with full pipeline
- ✅ Fast exponential approximation quality

### 2.5 What C1 would validate (that RoofGS has not)

- ✅ Backward gradient correctness in differentiable pipeline
- ✅ Collision rate / inversion analysis for IEEE 754 truncation
- ✅ Differentiable rendering compatibility
- ✅ Isolated sort-only speedup (not mixed with other optimizations)

---

## 3. SplatShop

### 3.1 Verified identity

**Paper:** Schütz et al. "Splatshop: Efficiently Editing Large Gaussian Splat Models." Computer Graphics Forum (Proc. HPG), 44(8), 2025.

**Type:** Interactive 3DGS **editing** toolbox (selection, deletion, painting, transformation)

**Source code:** https://github.com/m-schuetz/Splatshop

### 3.2 Sorting architecture (from source)

SplatShop has its own GPU sorting pipeline (**not CUB-based**):

- `src/GPUSorting/RadixSort.cu` — custom 16-bit radix sort implementation
- `src/GPUSorting/GPUSorting.h` — `sort_32bit_keyvalue()`, `sort_16bitkey_32bitvalue()`
- `src/GPUSorting/OneSweep.cu` — OneSweep sort variant

Depth-based sorting in `src/gaussians_rendering.cu`:
1. **Line 788:** `staging_depth[visibleSplatID] = depth` — store depth
2. **Lines 1005–1033:** Apply sort ordering to data arrays via `kernel_applyOrdering_*`
3. **Line 1037:** `kernel_createTilefragmentArray` — build tile fragment lists from depth-sorted splats
4. **Lines 1272–1295:** Tile boundary detection via `tileIDs` array

SplatShop's pipeline differs fundamentally from gsplat:
- **gsplat:** CUB global radix sort on `(tile_id | depth)` composite key (tile-major, depth-minor)
- **SplatShop:** Global depth sort first, then per-splat tile assignment → per-tile lists are implicitly depth-sorted

### 3.3 Relevance to C1

**Related only.** No evidence that SplatShop:
- Quantizes or truncates depth in the sort key
- Reduces sort key width for performance
- Measures collision from depth precision loss
- Uses a differentiable pipeline (edit-only)
- Discusses radix pass reduction as an optimization

**Conclusion:** SplatShop is NOT prior art for C1's mechanism. The sorting architectures are fundamentally different.

---

## 4. Faster-GS

### 4.1 Verified identity

**Paper:** Hahlbohm et al. "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization." arXiv 2602.09999, Feb 2026.

**Focus:** Training optimization (densification, gradient approximation, numerical stability, Gaussian truncation). Does **not** propose new sorting or depth key compression.

### 4.2 Relevance to C1

**Related only.** Faster-GS does **not**:
- Compress the depth sort key
- Quantize depth in the key
- Reduce CUB radix pass count
- Propose any sorting algorithm change

**Conclusion:** Faster-GS is NOT prior art for C1's mechanism.

---

## 5. Prior Works from Phase 17A Literature Recon

(Only works with verified evidence included.)

| Work | Evidence | Relevance to C1 |
|------|----------|-----------------|
| **Speedy-Splat** (Wang et al., 2024) | Confirmed Phase 17A | Orthogonal — visibility culling, tile dedup |
| **FlashGS** (Feng et al., 2024) | Confirmed Phase 17A | Orthogonal — tile scheduling, shared memory |
| **HiGS** (Hamdi et al., 2024) | Confirmed in gsplat source | Related — custom CTA-level 8-bit radix cascade, not key compression |
| **Taming 3DGS** (Muckley et al., 2024) | Confirmed | Related — notes int-depth quantization idea, **does not implement sort speedup** |
| **EVER** (Zheng et al., 2024) | Confirmed | Negative overlap — 96-bit key (wider, not narrower) |

---

## 6. C1 Classification

```
C1 Classification: B — Engineering adaptation, with elements of D — Concrete extension

Exact Prior Art:
  - RoofGS (arXiv 2608.15785) independently converges on the same core insight:
    "compress the depth sort key to reduce radix passes and memory traffic"

Main Overlap:
  - Both reduce sort key width to reduce radix passes
  - Both acknowledge collisions as acceptable trade-off
  - Both target the same pipeline bottleneck (memory-bound radix sort)

Main Difference:
  - Quantization method: IEEE 754 bitcast truncation (C1) vs uniform linear (RoofGS)
  - C1: 16-bit depth, 30-bit sort range, fixed bit allocation, no scene params
  - RoofGS: 19-bit depth (1080p/tile16), 32-bit key, resolution-adaptive, needs z_near/z_far
  - C1 is backward-pass compatible by design (5-line gsplat key-only change)
  - RoofGS is inference-only; does not address backward pass or differentiability
  - C1 isolates sort-only speedup; RoofGS bundles key compression with 4+ other optimizations

Potential Research Gap:
  - Differentiable depth key compression — C1 would be the first to demonstrate
    a truncated IEEE 754 depth key in a differentiable 3DGS pipeline.
  
  - Collision characterization for truncated depth — C1's Phase V2 provides
    empirical collision data. RoofGS quantifies collisions only as bounding
    statement: "depth difference smaller than quantization step."
  
  - Sort-only speedup decomposition — RoofGS reports end-to-end speedup with
    5+ simultaneous optimizations. C1 would isolate the sort-key effect alone.

  Note: C1's 30-bit sort range vs RoofGS's full 32-bit word means C1 starts
  from a narrower baseline (46 bits → 30 bits = 35% reduction) compared to
  RoofGS's baseline (64 bits → 32 bits = 50% reduction). Direct comparisons
  of pass-count reduction are misleading without accounting for different
  baselines and CUB policy.

Not Prior Art:
  - SplatShop: interactive editor with global depth pre-sort (not key compression)
  - Faster-GS: training optimization study (not sorting)
  - Speedy-Splat/FlashGS: orthogonal approaches

Recommendation: CONDITIONAL CONTINUE as differentiated empirical study
  - The specific quantization method (IEEE 754 >> 16) has no prior art
  - The differentiable backward validation has no prior art (all prior works
    are inference-only)
  - The isolated sort-only speedup measurement has no prior art
  - Proceed to minimal prototype to verify correctness before further claims
```

---

## 7. References

| # | Reference |
|---|-----------|
| 1 | **RoofGS:** Luo et al. "RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting." arXiv:2608.15785, Aug 2026. [arxiv.org/abs/2608.15785](https://arxiv.org/abs/2608.15785) |
| 2 | **TileGS:** Tan et al. "TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization." arXiv:2609.03613, 2026. |
| 3 | **SplatShop:** Schütz et al. "Splatshop: Efficiently Editing Large Gaussian Splat Models." Computer Graphics Forum (Proc. HPG), 44(8), 2025. [momentsingraphics.de/HPG2025](https://momentsingraphics.de/HPG2025.html), [github.com/m-schuetz/Splatshop](https://github.com/m-schuetz/Splatshop) |
| 4 | **Faster-GS:** Hahlbohm et al. "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization." arXiv:2602.09999, Feb 2026. |
| 5 | **gsplat:** Ye et al. "gsplat: An Open-Source Library for 3D Gaussian Splatting." GitHub: nerfstudio-project/gsplat, 2024. |
| 6 | **Phase 17A Literature Recon:** `reports/epic05/phase17a_literature_recon.md` |
| 7 | **C1 Source Audit Final:** `reports/phase-c17-c2/c1_source_audit_final.md` |
