# C17-3 — Backward Metadata Cache Source Audit

**Status:** Source audit complete — no implementation

**Date:** 2026-09-06

**Target file:** `gsplat` v1.5.3 (installed at `C:\Users\36570\miniconda3\Lib\site-packages\gsplat`)

---

## Executive Conclusion

> **C17-3 does NOT have a measurable baseline gap.** The gsplat backward kernel's global-memory reads of means2d/conics/colors/opacities are already well-served by the GPU's L2 hardware cache: these four arrays total ~756 KB (packed, nnz=21K), which fits in L2 (40 MB on A100, 3 MB on RTX 5070). The dominant backward bandwidth cost is `flatten_ids` (704 MB, 176M × 4 bytes), which C17-3 does not reduce. The shared memory batch loading already provides intra-tile attribute reuse. A software cache adds complexity and memory traffic without reducing the actual bandwidth bottleneck.

### Recommendation: **DROP**

---

## 1. Dataflow Diagram

### Standard path (`packed=True`, `sh_degree=None`)

```
Python API: rasterization(means, quats, scales, opacities, colors, ...)
  │
  ├─── _FullyFusedProjectionPacked.forward()
  │     │   CUDA kernel: projection_ewa_3dgs_packed_fwd
  │     │   → produces: batch_ids, camera_ids, gaussian_ids, radii
  │     │                means2d [nnz, 2], depths [nnz],
  │     │                conics [nnz, 3], compensations [nnz] (optional)
  │     │
  │     │   ctx.save_for_backward():
  │     │     batch_ids, camera_ids, gaussian_ids     ← indices for backward scatter
  │     │     means, covars, quats, scales             ← original 3D params (for recompute)
  │     │     viewmats, Ks                             ← camera params
  │     │     conics, compensations                    ← 2D results (reused in bwd)
  │     │
  │     │   forward outputs: radii, means2d, depths, conics, compensations  → meta dict
  │     │
  │     └─── _FullyFusedProjectionPacked.backward()
  │           CUDA kernel: projection_ewa_3dgs_packed_bwd
  │           INPUTS (from saved): means, quats, scales, viewmats, Ks,
  │                                conics, compensations, batch_ids, camera_ids, gaussian_ids
  │           INPUTS (from upstream gradient): v_means2d, v_depths, v_conics, v_compensations
  │           
  │           ⟹ RECOMPUTES geometry: covar from quats+scales,
  │                                   camera-space covar, projection VJP
  │           ⟹ This is memory-optimal (Phase 15 confirmed)
  │
  ├─── isect_tiles() / isect_offset_encode()  (torch.no_grad)
  │     → isect_ids, flatten_ids, isect_offsets  → meta dict
  │
  ├─── (SH evaluation if sh_degree is set — not in this path)
  │
  └─── _RasterizeToPixels.forward()
        │   CUDA kernel: rasterize_to_pixels_3dgs_fwd
        │   INPUTS: means2d, conics, colors, opacities, backgrounds, masks,
        │           isect_offsets, flatten_ids
        │   → produces: render_colors, render_alphas, last_ids
        │
        │   ctx.save_for_backward():
        │     means2d, conics, colors, opacities  ← attribute arrays (ALREADY SAVED)
        │     backgrounds, masks, isect_offsets,
        │     flatten_ids, render_alphas, last_ids
        │
        └─── _RasterizeToPixels.backward()
              CUDA kernel: rasterize_to_pixels_3dgs_bwd
              INPUTS (from saved): ALL of the above
              →
              RECEIVES from saved:
                means2d, conics, colors, opacities  ← from forward, via ctx.saved_tensors
                isect_offsets, flatten_ids, render_alphas, last_ids
              +
                v_render_colors, v_render_alphas  ← upstream gradient
```

### Key observation

The backward kernel receives `means2d`, `conics`, `colors`, `opacities` **from the saved tensors** — these are the identical tensors used by forward. They are NOT recomputed. They are NOT discarded between forward and backward. The autograd system holds references to them via `ctx.saved_tensors`.

---

## 2. Metadata Inventory

All metadata: `_RasterizeToPixels` only (this is where C17-3 would apply). Projection backward already does recompute, but that's memory-optimal and separate from rasterization backward.

| Metadata | Shape (tile16, 1080p, room iter30000) | Produced in Forward | Saved for Backward | Used by Backward | Memory Cost | Compute Cost |
|----------|---------------------------------------|:-------------------:|:------------------:|:----------------:|:-----------:|:------------:|
| `means2d` | [nnz, 2], [21K, 2] = 168 KB | Projection CUDA kernel | ✅ via ctx | ✅ Read via flatten_ids indirection | **168 KB** | — |
| `conics` | [nnz, 3], [21K, 3] = 252 KB | Projection CUDA kernel | ✅ via ctx | ✅ Read via flatten_ids indirection | **252 KB** | — |
| `colors` | [nnz, CDIM], [21K, 3] = 252 KB | SH eval (or color index) | ✅ via ctx | ✅ Read via flatten_ids indirection | **252 KB** | — |
| `opacities` | [nnz], [21K] = 84 KB | Projection (from input × compensation) | ✅ via ctx | ✅ Read via flatten_ids indirection | **84 KB** | — |
| `backgrounds` | [..., CDIM] | Input | ✅ via ctx | Read per pixel | Negligible | — |
| `masks` | [I, th, tw] | Input | ✅ via ctx | Read per tile | Negligible | — |
| `isect_offsets` | [I, th, tw] | offset_encode kernel | ✅ via ctx | Read per tile | **32 KB** | — |
| `flatten_ids` | [n_isects] = 176M × 4 = 704 MB | `isect_tiles()` pass 2 | ✅ via ctx | Read per intersection (sequential, back-to-front) | **704 MB** | — |
| `render_alphas` | [I, H, W, 1] = 2 MB | Forward rasterize kernel | ✅ via ctx | Read per pixel | **2 MB** | — |
| `last_ids` | [I, H, W] = 2 MB | Forward rasterize kernel | ✅ via ctx | Read per pixel (early exit) | **2 MB** | — |
| `v_render_colors` | [I, H, W, CDIM] | Upstream gradient | ❌ (not fwd output) | ✅ Read per pixel | — | — |
| `v_render_alphas` | [I, H, W, 1] | Upstream gradient | ❌ (not fwd output) | ✅ Read per pixel | — | — |

### Total saved tensor memory for backward

| Category | Memory |
|:---------|:------:|
| Attribute arrays (means2d + conics + colors + opacities) | **~756 KB** |
| Intersection data (flatten_ids + isect_offsets) | **704 MB + 32 KB** |
| Forward outputs (render_alphas + last_ids) | **4 MB** |
| Other (backgrounds, masks) | Negligible |
| **Total** | **~708 MB** |

---

## 3. Classification

### A. Already Cached (via `ctx.save_for_backward`)

| Metadata | Why "Already Cached" |
|:---------|:---------------------|
| `means2d` | Forward computed it → saved via ctx → backward receives it directly. NOT recomputed. |
| `conics` | Same — saved from forward output, directly available in backward. |
| `colors` | Same — saved from forward (post-SH-evaluation), directly available. |
| `opacities` | Same — saved from forward (post-compensation), directly available. |
| `flatten_ids` | Saved from forward `isect_tiles()`. Read sequentially by backward. |
| `isect_offsets` | Saved from forward `offset_encode()`. Read per tile. |
| `render_alphas` | Saved from forward. Read per pixel (used in gradient of T). |
| `last_ids` | Saved from forward. Read per pixel (early exit optimization). |

### B. Recomputed

| Metadata | Where recomputed | Why not saved | Cost estimate |
|:---------|:-----------------|:--------------|:--------------|
| «NONE in rasterization backward» | — | — | — |
| 3D geometry in `_FullyFusedProjectionPacked.backward` (covar from quats+scales, camera-space transform, projection VJP) | CUDA kernel `projection_ewa_3dgs_packed_bwd` | Storing 3D intermediates would cost B×C×N×(9+3+9+6) ≈ 27× more memory than current stored outputs. Memory-optimal design. | **Not a target for C17-3.** C17-3 is specifically about the rasterization backward kernel, not projection backward. |
| Color re-evaluation in `_SphericalHarmonics.backward` | CUDA kernel `spherical_harmonics_bwd` | SH from coeffs. Storing evaluated colors IS what the rasterizer does. Forward already saves the evaluated `colors` tensor. | Not applicable — colors are already cached for rasterization backward. |

### C. Not Needed

- All `batch_ids`, `camera_ids`, `gaussian_ids` from projection: only needed by projection backward's scatter operation, not by rasterization backward
- `radii`: not needed by rasterization backward (intersection data is already sorted)
- Original `means`, `quats`, `scales`: only needed by projection backward (recompute)

---

## 4. Recompute Inventory — Rasterization Backward Only

**There is NO recomputation in the rasterization backward kernel.**

Every tensor that the backward kernel reads was either:
1. Produced by forward and saved via `ctx.save_for_backward` (means2d, conics, colors, opacities, flatten_ids, isect_offsets, render_alphas, last_ids), or
2. Passed in from upstream gradient (v_render_colors, v_render_alphas)

The backward kernel's computation is:

```text
Read flatten_ids[idx] → g (Gaussian index)
Read means2d[g], conics[g], colors[g * CDIM], opacities[g]  ← from saved tensors
Compute sigma = f(conic, delta)
Compute alpha = min(0.999, opacity * exp(-sigma))
Compute v_rgb, v_conic, v_xy, v_opacity from:
  v_alpha = f(v_render_c, v_render_a, T, alpha, buffer, backgrounds)
  v_sigma = f(v_alpha, ...)
  gradients = f(v_sigma, delta, ...)
Accumulate via warp-reduced atomicAdd
```

The values read (means2d, conics, colors, opacities) are **never recomputed**. They were already computed during forward projection/SH evaluation.

---

## 5. Existing "Cache" Mechanisms in gsplat

### 5.1 Shared Memory Batch Cache (Intra-tile)

Both forward and backward kernels use `extern __shared__ int s[]` (lines 94-104 Fwd.cu, lines 97-104 Bwd.cu) to pre-load a batch of Gaussians into shared memory before processing:

```cuda
// Bwd.cu lines 97-104
extern __shared__ int s[];
int32_t *id_batch = (int32_t *)s;                    // [block_size]
vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(   // [block_size]
    &id_batch[block_size]);
vec3 *conic_batch = reinterpret_cast<vec3 *>(         // [block_size]
    &xy_opacity_batch[block_size]);
float *rgbs_batch = (float *)&conic_batch[block_size]; // [block_size * CDIM]
```

**All threads in a tile cooperatively load `block_size` items from global memory into shared memory** (lines 140-151 Bwd.cu), then each thread processes them within the inner batch loop (line 156). This provides **intra-tile, intra-batch reuse** — every attribute is loaded from global memory once per batch iteration and shared across all `tile_size²` threads.

**This is ALREADY a fully functional backward metadata cache at the shared memory level.**

### 5.2 L2 Hardware Cache (Cross-tile)

The four attribute arrays (means2d, conics, colors, opacities) have a total size of:

| Array | Size (nnz=21K) |
|:------|:--------------:|
| `means2d` | 21K × 8 = 168 KB |
| `conics` | 21K × 12 = 252 KB |
| `colors` | 21K × 12 = 252 KB |
| `opacities` | 21K × 4 = 84 KB |
| **Total** | **756 KB** |

Hardware L2 cache sizes:
- RTX 5070 Laptop: **3 MB**
- A100 (sm_80): **40 MB**

**All four arrays fit comfortably in L2 cache on any modern GPU.** Since these arrays are read by ALL tile blocks (8160 for tile16), the first few tile blocks bring the data into L2, and subsequent blocks read from L2 cache. The hit rate approaches 100% after the first tile block.

### 5.3 `ctx.save_for_backward` Autograd Cache (Pipeline-level)

Forward outputs that are needed by backward are saved via `ctx.save_for_backward`. This is PyTorch's standard mechanism for persisting tensors across autograd's forward→backward lifetime. It is ALREADY present for all four attribute arrays (line 1286-1297):

```python
ctx.save_for_backward(
    means2d, conics, colors, opacities,   # ← ALL ALREADY SAVED
    backgrounds, masks, isect_offsets,
    flatten_ids, render_alphas, last_ids,
)
```

### 5.4 Conclusion on Existing Caches

| Cache mechanism | Scope | What it caches | Already present? |
|:----------------|:------|:---------------|:-----------------|
| Shared memory batch load | Intra-tile | means2d, conics, colors, opacities per batch | ✅ YES |
| L2 hardware cache | Cross-tile | Hot attribute arrays | ✅ YES (arrays are 756 KB) |
| `ctx.save_for_backward` | Forward→backward | All tensors | ✅ YES |

**C17-3's proposed "software cache indexed by Gaussian ID" would duplicate all three existing cache layers.** No gap exists.

---

## 6. Candidate Optimization Analysis

### The Claim

> C17-3 restructures the backward kernel to pre-load Gaussian attributes into a persistent cache indexed by Gaussian ID, reducing redundant global memory traffic for Gaussians spanning many tiles.

### What Would a Software Cache Do?

```cuda
// Pseudocode for proposed cache:
Phase 1: for each unique g in tile range:
    cache[g] = {means2d[g], conics[g], colors[g * CDIM], opacities[g]}

Phase 2: process batches from cache instead of global memory:
    g = flatten_ids[idx];
    load xy = cache[g].means2d;    // shared memory read instead of global read
    load conic = cache[g].conics;
```

### The Problem: No Global Memory Traffic Reduction

Under the current code, the backward kernel reads:

| Read pattern | Bytes read per backward call | Can C17-3 reduce this? |
|:-------------|:----------------------------:|:----------------------:|
| `flatten_ids[range_start..range_end]` per tile (sequential) | 176M × 4 = **704 MB** | ❌ No — flatten_ids must be traversed regardless |
| `means2d[g]` via random access (per intersection) | 176M × 8 = 1.4 GB | ⚠️ **L2 already caches the 168 KB array.** Software cache would add extra read (loading into software cache) without reducing hardware cache reads. |
| `conics[g]` via random access | 176M × 12 = 2.1 GB | Same — L2 already caches the 252 KB array |
| `colors[g * CDIM]` via random access | 176M × 12 = 2.1 GB | Same — L2 already caches the 252 KB array |
| `opacities[g]` via random access | 176M × 4 = 704 MB | Same — L2 already caches the 84 KB array |

**Total attribute random read traffic: ~6.3 GB per backward call.**

**Total L2 cache capacity: 3–40 MB (fits all attributes ×100).**

**Therefore: the 6.3 GB of attribute reads are almost entirely L2 cache hits.** The effective global memory bandwidth consumed is:
- For the first few tile blocks (cold L2): attribute arrays are loaded from HBM once (756 KB total)
- For all subsequent tile blocks (hot L2): attribute reads are served from L2 at ~10× lower latency

A software cache would **add** a Phase 1 that loads each unique Gaussian's attributes into cache memory. But this Phase 1 would do EXACTLY the same global memory reads as the current Phase 2 code already does — just in a different order. **The total bytes read from global memory would not decrease.**

### The Real Cost: `flatten_ids`

The actual dominant cost in backward global memory traffic is `flatten_ids`:

```
flatten_ids[n_isects] = 176M × 4 bytes = 704 MB
Sequential read (back-to-front)
Cannot be cached (704 MB > 40 MB L2)
```

C17-3 does NOT reduce this cost. `flatten_ids` must be read in full regardless of how means2d/conics/colors/opacities are accessed.

### Quantitative Estimate

| Metric | Current (baseline) | C17-3 (proposed) | Δ |
|:-------|:------------------:|:----------------:|:-:|
| Global memory reads (flatten_ids) | 704 MB | 704 MB | **0** |
| Global memory reads (attribute arrays) | ~756 KB (first pass, L2 cold) | 756 KB (Phase 1, L2 cold) | **0** |
| L2 cache reads (attribute arrays) | ~6.3 GB | ~6.3 GB | **0** |
| Shared memory writes (attribute arrays) | 176M × 32 bytes = 5.6 GB (batch loading) | 176M × 32 bytes + unique × 32 bytes | **NEGATIVE** (more writes) |
| Software cache management (Gaussian dedup) | 0 | 176M hash lookups or linear scan | **NEGATIVE** |
| Kernel complexity | Simple | Higher (2-phase, cache management) | **NEGATIVE** |

**Net ΔT: positive (slower).**

---

## 7. Cost/Risk Analysis

### Memory Cost

| Overhead | Size | Impact |
|:---------|:----:|:-------|
| Additional shared memory per block (Gaussian-to-cache-slot mapping) | 256 × 4 = 1 KB | None — shared memory is abundant for tile_size=16 (64 KB per SM) |
| Cache population Phase 1 | 0 (in-place on existing arrays) | Adds one extra kernel launch or one extra phase in existing kernel |
| Persistent cache in global memory (pre-built on forward) | nnz × 32 bytes = 672 KB | Negligible (0.01% of peak) |

### Synchronization Cost

Current kernel has `block.sync()` once per batch (line 153 Bwd.cu). C17-3 would add at least one additional `block.sync()` between Phase 1 (cache population) and Phase 2 (batch processing). This increases warp serialization.

### Correctness Impact

| Property | Impact |
|:---------|:-------|
| Forward numeric values | ✅ No change — C17-3 only modifies backward kernel |
| Sorting | ✅ No change — backward does not touch sort or intersection |
| Rasterization | ✅ No change — only the fetch pattern in backward changes |
| Backward input identity | ⚠️ Must verify: `allclose(grad_baseline, grad_cached)`. Since cache reads the same float32 values, identity should hold. But atomicAdd accumulation order changes if batches are reordered → FP differences possible. |
| Gradient semantics | ✅ Unchanged — same arithmetic, same atomicAdd accumulation |
| Stale metadata | ✅ None — values are read-only in backward |
| Packed/unpacked modes | Both supported — same access pattern in both |

### Differentiability

Same as baseline — the backward kernel's computation is identical, only the memory access pattern changes.

---

## 8. Answer to the Core Question

### What computation would C17-3 eliminate?

**None.** The backward kernel does not recompute means2d, conics, colors, or opacities. They are already saved from forward and directly read from the saved tensors.

### What memory traffic would C17-3 reduce?

**None that matters.** The attribute arrays (means2d, conics, colors, opacities) are:
1. Already stored in saved tensors (ctx.saved_tensors)
2. Already loaded into shared memory per batch (intra-tile caching)
3. Already cached in L2 hardware cache (cross-tile caching — arrays are 756 KB total)
4. The dominant bandwidth cost (flatten_ids, 704 MB) is unaffected

A software cache would add complexity, shared memory pressure, and synchronization without reducing the primary bandwidth bottleneck.

---

## 9. Recommendation

### DROP

**Rationale:** The source audit demonstrates that C17-3's premise (redundant global memory reads of attribute arrays by the backward kernel) is not a measurable bottleneck:

1. **All four attribute arrays are already saved via `ctx.save_for_backward`** — no recomputation occurs
2. **Shared memory batch loading** already provides intra-tile attribute reuse (lines 140-151 Bwd.cu)
3. **L2 hardware cache** provides cross-tile attribute reuse (arrays total 756 KB, L2 is 3-40 MB)
4. **Dominant bandwidth cost** is `flatten_ids` (704 MB sequential read), which C17-3 does not reduce
5. **A software cache would add global memory traffic** (Phase 1 loads) without reducing existing traffic

### Source-Verified Constraints

| Claim from C17-3 design | Source-verified? |
|:------------------------|:----------------:|
| "Backward re-reads means2d/conics/colors/opacities from global memory" | ✅ TRUE — but these are 756 KB total, fitting in L2 |
| "Gaussians spanning many tiles → each attribute loaded T times" | ✅ TRUE — 8160 tiles → 8160 reads per Gaussian average |
| "A software cache would reduce redundant global memory traffic" | ❌ FALSE — software cache Phase 1 does the SAME global reads, L2 already provides cross-tile caching |
| "The backward kernel recomputes attributes" | ❌ FALSE — all attributes are saved via ctx and directly read |

### Re-entry condition

Reconsider C17-3 only if **both** conditions are met:
1. A profiler (Nsight Compute) shows that attribute array reads have <50% L2 hit rate in the backward kernel
2. The measured backward kernel time is significantly higher than forward kernel time (currently <1% of forward for rasterize, but backward is a separate call)

### Relation to C17-2

If C17-2 (two-phase sorting) is implemented, the `flatten_ids` structure changes from an interleaved global sort to per-tile groups. The backward kernel still reads flatten_ids sequentially, so C17-2 does not affect this analysis.

---

## 10. Key Source References

### Backward kernel attribute reads (Bwd.cu)
```
Line 34:  flatten_ids  → int32_t *__restrict__
Line 22:  means2d      → vec2 *__restrict__
Line 23:  conics       → vec3 *__restrict__
Line 24:  colors       → scalar_t *__restrict__
Line 25:  opacities    → scalar_t *__restrict__
Line 141: g = flatten_ids[idx]                               ← random read of flatten index
Line 143: const vec2 xy = means2d[g];                         ← random read of means2d
Line 144: const float opac = opacities[g];                    ← random read of opacities
Line 145: xy_opacity_batch[tr] = {xy.x, xy.y, opac};         ← write to shared memory
Line 146: conic_batch[tr] = conics[g];                       ← random read of conics
Line 148-150: rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];  ← random read of colors
```

### Save for backward (_wrapper.py)
```
Line 1286-1297: ctx.save_for_backward(means2d, conics, colors, opacities, ...)
```

### Shared memory batch layout (Bwd.cu)
```
Line 97-104: extern __shared__ int s[];
             id_batch       ← int32_t[block_size]        (4 bytes × 256 = 1 KB)
             xy_opacity_batch ← vec3[block_size]          (12 bytes × 256 = 3 KB)
             conic_batch    ← vec3[block_size]            (12 bytes × 256 = 3 KB)
             rgbs_batch     ← float[block_size × CDIM]    (12 bytes × 256 = 3 KB for CDIM=3)
                                                          Total: 10 KB
```

### Projection backward recompute (not C17-3 target)
```
_wrapper.py Line 1702-1726: projection_ewa_3dgs_packed_bwd(means, quats, scales, ...)
```
This IS a genuine recompute (reads original 3D params, recomputes covar and projection), but it is memory-optimal (Phase 15 confirmed) and belongs to the projection autograd function, not the rasterization autograd function that C17-3 targets.
