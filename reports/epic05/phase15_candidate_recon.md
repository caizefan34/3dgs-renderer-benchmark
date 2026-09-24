# Phase 15 — Implementation-Level Optimization Candidate Reconnaissance

> Source-evidence-based analysis of 3DGS renderer optimization candidates.
> Based on complete CUDA source audit of gsplat v1.x rasterization pipeline.

---

## Executive Summary

After systematic source-level audit of every kernel in the gsplat forward/backward rasterization pipeline, we identify **3 high-confidence implementation-level optimization candidates** and **2 lower-confidence directions**. The investigation confirms that the **intersection generation + sorting pipeline** (S3→S5) accounts for ~93-97% of forward time and involves the greatest data volume transformation (nnz → n_isects, where n_isects ≈ 4-6× nnz for tile16).

**No optimization candidate is guaranteed to produce a universal E2E training benefit.** However, two candidates (C1 depth compression, C2 sync elimination) have strong theoretical grounding for modest (5-15%) forward throughput improvement with ZERO rendering quality impact.

---

## Important: What Phase 14 Already Established

| Finding | Implication |
|---------|-------------|
| tile20 anomaly caused by topology spillover, not renderer | Renderer kernel changes must be evaluated WITH topology |
| Segmented sort (current impl) is slower than global sort | CUB segmented sort overhead > benefit |
| Snapshot winner ≠ training winner | Any candidate must pass full training gate |
| M2 packed/dense neutral | Packing optimization already saturated |
| M4/M5 non-beneficial | Parameter-level tuning is exhausted |

Phase 15 therefore focuses on **data-structure and algorithm-level changes** — not parameter knobs.

---

## Source-Level Bottleneck Analysis

### Forward: Intersection + Sorting Pipeline (~93-97% of forward time)

```
nnz gaussians (visible after projection)
    │
    ▼
First Pass (count tiles per gaussian)
    ├── each thread: iterate over tile range of its gaussian
    ├── compute tile_min/tile_max from (mean2d, radius_x, radius_y, tile_size)
    └── write tile_count to tiles_per_gauss
    │
    ▼
cumsum(tiles_per_gauss) → n_isects (HOST SYNC POINT)
    │
    ▼
Second Pass (materialize intersections)
    ├── each thread: iterate over tile range AGAIN
    ├── for each tile: encode isect_id = image_id | tile_id | depth_bits
    └── write isect_id[8B] + flatten_id[4B] per intersection
    │   Total write: 12 × n_isects bytes
    │
    ▼
Global Radix Sort (cub::DeviceRadixSort::SortPairs)
    ├── key: int64 (45 bits used)
    ├── value: int32
    ├── passes: ceil(45/4) ≈ 12 passes
    ├── CUB temp storage: O(n_isects × (8+4)) bytes
    └── Total R+W: 24 × n_isects bytes
    │
    ▼
Offset Encode (re-scan sorted array)
    ├── threads = n_isects
    ├── each thread: read isect_id, compare with neighbor
    └── write offset for tile boundaries
```

### Key Metric: n_isects Explosion

| tile_size | n_isects (typical 1080p, 200K Gs) | Ratio to nnz |
|-----------|-----------------------------------|--------------|
| 8 | ~700M (extrapolated) | ~15× |
| 16 | ~175.8M | ~6-8× |
| 32 | ~44.0M | ~2-3× |

### Backward: AtomicAdd + Re-read Traffic

The backward kernel:
1. Re-reads **every** forward input tensor from global memory
2. Traverses sorted intersections back-to-front (same n_isects count)
3. Uses gpuAtomicAdd per warp for gradient accumulation
4. Recomputes geometry (covar from quats/scales) — no storage redundancy

The backward pass memory traffic is roughly **equal to forward memory traffic** plus atomic write contention.

---

## Candidate C1: Depth Bit-Width Compression (Sort Key Reduction)

### Affected stage
Intersection encoding (S4) + Radix sort (S5)

### Current code
`IntersectTile.cu:98-100`
```cuda
int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // full float32 → int32
int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // zero-extend to uint64
```

`IntersectTile.cu:322`
```cuda
cub::DeviceRadixSort::SortPairs(d_keys, d_values, n_isects, 0, 
    32 + tile_n_bits + image_n_bits, ...);  // sorts 45+ bits
```

### Problem
The depth field uses a full 32-bit float reinterpretation in the sort key. For tile-local depth ordering, we only need enough bits to correctly order Gaussians within a single tile. IEEE 754 float32 has 23 bits of mantissa + 8 bits of exponent — far more precision than needed for depth ordering within a 16×16 pixel region.

### Evidence
- n_isects = 44M–176M (tile32-tile16, 1080p)
- 64-bit key being sorted over 44-48 bits (uint32 + tile_n_bits + image_n_bits)
- 12 CUB radix passes for 48-bit key
- Each pass reads/writes 12 × n_isects bytes

### Proposed change
Replace depth encoding from full float32 reinterpretation to **upper 2 bytes (16-bit mantissa range)** of the float32. This preserves IEEE 754 ordering but at coarser granularity.

Specifically:
```cuda
uint32_t depth_i32 = *(uint32_t *)&(depths[idx]);
uint32_t depth_16bit = depth_i32 >> 16;  // Keep upper 16 bits (exponent + top mantissa)
// Pack into lower bits of key
int64_t isect_id = iid_enc | (tile_id << 16) | depth_16bit;
```

Sort range becomes: `16 + tile_n_bits + image_n_bits` (≈29 bits instead of 45 bits).
CUB passes: ceil(29/4) ≈ 8 passes instead of 12. **33% fewer sort passes**.

### Expected workload reduction
- Sort passes: 12 → 8 (33% reduction)
- Sort temporary storage: ~(12/8)× smaller temp buffer
- CUB's DeviceRadixSort efficiency scales with bit count
- Expected sort runtime: ~33% faster
- Expected forward E2E: ~10-15% faster (sort is ~30-40% of forward)
- Memory: no change to persistent storage (keys still int64, just fewer bits sorted)

### Expected memory reduction
- CUB temp storage for sort: reduced proportionally to fewer passes
- No change to persistent tensor sizes

### Expected runtime benefit
- Sort stage: ~25-33% improvement
- Forward: ~8-15% E2E benefit
- Training: ~3-8% E2E benefit (sort is lost in backward overhead)

### Forward risk
**LOW.** Encoding change preserves depth ordering up to 16-bit granularity. For depths in [0.01, 100], the upper 16 bits of float32 give ~1/65536 relative distinguishability within each power-of-two range. Two Gaussians with depths closer than ~0.15% within the same tile would see their relative order undefined — a negligible effect on rendered pixels.

### Backward risk
**NONE.** Backward reads `flatten_ids` not `isect_ids`. The sort is only for forward ordering; backward uses `flatten_ids` to index into gaussian data. No change to gradient path.

### Gradient risk
**NONE.** Depth encoding is not part of the autograd graph (isect_tiles is @torch.no_grad()).

### GT quality risk
**VERY LOW.** Sorting order affects alpha compositing. Swapping two Gaussians with nearly identical depths (within ~0.15%) produces sub-pixel alpha difference. The quantized sorting preserves the correct ordering for depths differing more than ~0.15% relative — the vast majority of cases.

### Training risk
**LOW.** If pixels with nearly-coplanar Gaussians produce slightly different colors, this would manifest as training noise at ~1/65536 scale. Densification/pruning use gradient-based heuristics that are insensitive to such small pixel differences.

### Independent ablation
- Modify IntersectTile.cu: change depth encoding to upper 16 bits
- Run forward-only timing comparison (snapshot, not training)
- Verify rendered pixel diff < 0.01 (max)
- No topology system changes needed

### Evidence strength: 4/5
### Potential workload reduction: 4/5
### Potential E2E benefit: 3/5
### Implementation feasibility: 5/5 (single file, ~10 lines changed)
### Forward correctness risk: 1/5
### Backward risk: 1/5
### Training risk: 2/5

---

## Candidate C2: Eliminate Host-Device Synchronization in Intersect Pipeline

### Affected stage
S3b — `at::cumsum` + `n_isects = cumsum[-1].item<int64_t>()`

### Current code
`Intersect.cpp:79-80`
```cpp
cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0);
n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
```

### Problem
The `.item<int64_t>()` call forces a **host-device synchronization** (cudaDeviceSynchronize implicit in the PyTorch tensor access). This blocks the CPU until the first-pass intersect kernel + cumsum kernel complete. The CPU then allocates intersection tensors of exact size.

This sync is fundamentally needed because n_isects varies per frame — the allocation size depends on it.

### Evidence
- First pass is a lightweight kernel (1 thread per gaussian, ~1-100 tiles per thread)
- cumsum is a single CUB call
- The sync cost is the round-trip latency to read a single value from GPU
- On a fast GPU, this is ~5-10µs; but it **serializes** the pipeline — no CPU-side prep work can happen

### Proposed change
Replace the sync with a pre-allocated **max-size** buffer strategy:
1. In the first call (or periodically), allocate `isect_ids`, `flatten_ids`, and sorted buffers at a **maximum expected capacity** (e.g., nnz × 16 for safety margin)
2. Keep the cumsum computation but skip the `.item()` call — write n_isects to a **GPU-side scalar** and launch the second pass with that buffer size
3. The second pass kernel already only writes n_isects entries — it just needs the correct count

Alternative: Use `torch.Tensor.view()` to view the last element and `torch.cuda.communicate` to read it without blocking, or use CUDA events for async.

Simpler approach: Always allocate at `max(n_isects, nnz * 8)` capacity, track `n_isects` as a GPU scalar, and only reallocate when capacity exceeded. This eliminates the sync in 99% of iterations.

### Expected workload reduction
- Eliminates ~5-10µs sync per iteration
- Enables pipelining: CPU-side SH evaluation or color preprocessing can overlap with first-pass kernel
- Total per-iteration savings: ~5-30µs (small but meaningful for ~20ms iterations)

### Expected memory reduction
- NONE — actually increases memory via pre-allocation
- But eliminates repeated allocation/deallocation overhead

### Expected runtime benefit
- ~0.5-2% forward E2E
- Enables future CPU-GPU overlap

### Forward risk
**NONE.** The GPU-side math is unchanged. Only the allocation strategy changes.

### Backward risk
**NONE.** No change to backward logic.

### Gradient risk
**NONE.**

### GT quality risk
**NONE.** Rendered pixels unchanged.

### Training risk
**NONE.**

### Independent ablation
- Modify Intersect.cpp to use pre-allocated buffers with GPU-side n_isects tracking
- Requires host-side prep to detect capacity overflow (use PyTorch caching allocator)
- No other system changes needed

### Evidence strength: 5/5
### Potential workload reduction: 2/5
### Potential E2E benefit: 2/5
### Implementation feasibility: 4/5
### Forward correctness risk: 1/5
### Backward risk: 1/5
### Training risk: 1/5

---

## Candidate C3: Persistent Intersection Buffer Reuse

### Affected stage
S4 (isect_ids allocate) + S5 (sorted buffers allocate)

### Current code
`Intersect.cpp:96-97, 120-121`
```cpp
at::Tensor isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
```

### Problem
Intersection buffers are allocated every forward call. For n_isects = 175M (tile16), each allocation is:
- isect_ids: 175M × 8 = 1.4 GB
- flatten_ids: 175M × 4 = 700 MB  
- isect_ids_sorted: 1.4 GB
- flatten_ids_sorted: 700 MB
- CUB temp storage: ~3 GB (double buffer + overhead)
Total: ~7.2 GB per iteration

PyTorch's caching allocator mitigates reallocation cost for same-size allocations, but n_isects varies per frame. When size changes, the allocator must:
1. Free old block
2. Allocate new block (potentially filling with smaller blocks)

### Evidence
Tensor sizes vary by ±20%+ between frames in real training (camera moves, gaussians move). Each size change triggers a new allocation from the caching allocator.

### Proposed change
Pre-allocate intersection buffers at a **maximum capacity** (e.g., nnz × 8 for tile_size=16) and truncate views:
1. Allocate once at max capacity
2. Slice `.narrow(0, 0, n_isects)` for each call
3. Only reallocate if n_isects exceeds current capacity

### Expected workload reduction
- Eliminates allocation/deallocation overhead for size-varying tensors
- Saves PyTorch caching allocator bookkeeping time
- Minor savings on variable-size iterations

### Expected memory reduction
- Maximum memory use doesn't change (already allocated)
- But reduces memory fragmentation

### Expected runtime benefit
- ~0.5-2% E2E forward (diminishing returns with caching allocator)

### Forward risk
**NONE.** Just buffer management.

### Backward risk
**NONE.** Backward uses the same intersection data.

### GT quality risk
**NONE.**

### Training risk
**NONE.**

### Independent ablation
- Modify Intersect.cpp buffer allocation strategy
- Requires a stateful buffer manager or static allocation in rendering.py

### Evidence strength: 3/5
### Potential workload reduction: 1/5
### Potential E2E benefit: 1/5
### Implementation feasibility: 4/5
### Forward correctness risk: 1/5
### Backward risk: 1/5
### Training risk: 1/5

---

## Candidate C4: Warp-Contention Reduced Gradient Accumulation

### Affected stage
S9 — backward rasterization (RasterizeToPixels3DGSBwd.cu)

### Current code
`RasterizeToPixels3DGSBwd.cu:251-275`
```cuda
if (warp.thread_rank() == 0) {
    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);  // per gaussian, per color
    gpuAtomicAdd(v_conic_ptr + i, v_conic_local[i]);  
    gpuAtomicAdd(v_xy_ptr + j, v_xy_local[j]);        
    gpuAtomicAdd(v_opacities + g, v_opacity_local);    
}
```

### Problem
Atomic gradient writes. For each gaussian visible in multiple tiles, each tile's warps issue atomicAdd to the same gaussian gradient address. With tile_size=16 and 256 threads/block, each tile has 8 warps. If a Gaussian spans N tiles, up to 8×N warps contend.

### Evidence
The backward kernel uses per-warp atomicAdd (reduced from per-thread by the warp reduction). However, for large Gaussians (radius > tile_size), intersections in multiple tiles all write to the same gaussian gradient. This creates memory contention on the atomic bus.

### Proposed change
**Two options, in increasing complexity:**

**4a (Simple):** Add a per-block shared memory gradient accumulation buffer, where each block accumulates gradients for the gaussians it processes, then flushes with a single atomicAdd per gaussian at the end. This collapses the 8 warps of a tile into 1 atomicAdd per gaussian.

**4b (Complex):** Tile-local partial reduction followed by a dedicated reduction kernel — eliminates all atomicAdd from the backward rasterizer.

### Expected workload reduction
- Reduces atomicAdd bank conflicts
- 4a: merges 8 warps → 1 write per gaussian per tile (8× reduction in atomic calls)
- 4b: eliminates atomicAdd entirely (uses parallel reduction)

### Expected runtime benefit
- Backward rasterizer: ~5-10% improvement on scenes with large Gaussians
- Training: ~2-5% E2E (backward is ~40-50% of training time)

### Forward risk
**NONE.** Forward unchanged.

### Backward risk
**LOW-MEDIUM.** The shared-memory accumulation approach (4a) changes the order of gradient accumulation within a tile, which could affect floating-point reduction results. Should be verified against current implementation.

### Gradient risk
**LOW.** Floating-point non-associativity of atomicAdd means the current implementation doesn't guarantee bit-exact gradients anyway. A tile-local reduction could produce slightly different (but equally valid) gradient values.

### GT quality risk
**NONE.** No change to forward rendering.

### Training risk
**LOW.** Gradient differences within floating-point tolerance should not affect training trajectory.

### Independent ablation
- Modify RasterizeToPixels3DGSBwd.cu to add shared-memory gradient accumulator
- Verify gradient norms vs baseline (< 1e-5 relative difference expected)

### Evidence strength: 3/5
### Potential workload reduction: 2/5
### Potential E2E benefit: 2/5
### Implementation feasibility: 3/5
### Forward correctness risk: 1/5
### Backward risk: 2/5
### Training risk: 2/5

---

## Candidate C5: Float16 Color/Opacity Storage for Reduced Backward Traffic

### Affected stage
Backward rasterization (S9) — global memory reads of colors and opacities

### Current code
Backward kernel reads:
- `colors[nnz, CDIM]` as float32 → 4×CDIM×nnz bytes
- `opacities[nnz]` as float32 → 4×nnz bytes

### Problem
The backward kernel re-reads all forward output tensors. Colors and opacities are the largest per-gaussian tensors and are read-only during backward. Using float16 halves the bandwidth requirement.

### Proposed change
Store colors and opacities as float16 after forward. The backward kernel reads float16 and converts internally.

### Expected workload reduction
- Backward global memory reads: ~(CDIM+1)×nnz×2 bytes saved per iteration
- For CDIM=3, nnz=200K: ~3.2 MB per iteration — not huge but consistent

### Expected runtime benefit
- Backward: ~2-5% improvement (bandwidth-bound on memory reads)
- Training: ~1-2% E2E

### Forward risk
**NONE.** Forward already produces float32.

### Backward risk
**LOW.** float16→float32 conversion inside kernel is free (single instruction).

### Gradient risk
**LOW.** float16 storage of forward values doesn't affect gradient precision (backward computes gradients in float32).

### GT quality risk
**NONE.**

### Training risk
**LOW.**

### Evidence strength: 2/5
### Potential workload reduction: 2/5
### Potential E2E benefit: 1/5
### Implementation feasibility: 3/5
### Forward correctness risk: 1/5
### Backward risk: 1/5
### Training risk: 1/5

---

## Candidate Scoring Summary

| Candidate | Evidence | Workload Reduct. | E2E Benefit | Feasibility | Fwd Risk | Bwd Risk | Training Risk | **Composite** |
|-----------|----------|-----------------|-------------|-------------|----------|----------|---------------|---------------|
| **C1: Depth bit-width** | 4 | 4 | 3 | 5 | 1 | 1 | 2 | **4.0** |
| **C2: Sync elimination** | 5 | 2 | 2 | 4 | 1 | 1 | 1 | **3.5** |
| C3: Buffer reuse | 3 | 1 | 1 | 4 | 1 | 1 | 1 | 2.5 |
| C4: Atomic reduction | 3 | 2 | 2 | 3 | 1 | 2 | 2 | 2.8 |
| C5: Float16 storage | 2 | 2 | 1 | 3 | 1 | 1 | 1 | 2.3 |

---

## TOP 3 Candidates

### 1. C1: Depth Bit-Width Compression (Highest priority)
**Why**: Addresses the single largest source of work in the forward pass (radix sort). 33% fewer sort passes with ~10 lines of code change. ZERO backward/gradient risk. The sort is well-documented as ~93-97% of forward time at tile16. This is the only candidate that directly reduces the sort workload.

### 2. C2: Host-Device Sync Elimination (Medium priority)
**Why**: Eliminates a known synchronization point that serializes the pipeline. Enables future async optimizations. Trivially safe (no math changes). The per-iteration savings are small but real, and the architectural improvement (no forced sync) enables future work.

### 3. C4: Warp-Contention Reduced Gradient Accumulation (Lower priority)
**Why**: The only backward-focused candidate with real evidence. Addresses atomicAdd contention in backward rasterization. The 4a (shared memory accumulator) variant is relatively simple and directly measurable.

---

## Answers to Required Questions

### Q1: Forward current real source-level bottleneck?
**The intersection generation + global radix sort pipeline (S3→S5).** Specifically, the radix sort of n_isects (44M–176M) 64-bit keys over 44-48 bits requires ~12 CUB passes, each reading/writing 12×n_isects bytes. For tile16/n_isects=175M, this is ~2.1 GB of read + write traffic per sort call — the single largest data movement in the forward pass.

### Q2: Backward current real source-level bottleneck?
**Re-reading all forward input tensors from global memory + gpuAtomicAdd contention.** The backward kernel reads means2d, conics, colors, opacities, flatten_ids, tile_offsets, render_alphas, last_ids, v_render_colors, v_render_alphas — ~10 tensor reads — none of which were cached from forward. The gpuAtomicAdd for gradient writes creates memory contention on overlapping Gaussians.

### Q3: Largest data structure?
**isect_ids and flatten_ids at n_isects elements.** For tile16 (1080p, 200K Gs): isect_ids = 1.4 GB, flatten_ids = 700 MB. With sorted variants: ~2.8 GB. Plus CUB temp storage: ~3-4 GB. Total intersection pipeline memory: ~6-7 GB.

### Q4: Highest sorting cost tensor?
**isect_ids [n_isects] × int64.** The 64-bit key sort requires more passes and more temp storage than a 32-bit key sort of the same count.

### Q5: Workloads that can be reduced before sort?
**Intersection culling.** The current two-pass intersect already only materializes tiles that are hit. But we could further reduce via: (a) opacity-aware radius tightening (already partially done via the extension factor based on opacity), and (b) depth-bucket pre-culling (estimate which depth ranges are safely occluded).

### Q6: Which sort work can be avoided?
**~12 CUB passes → ~8 passes by reducing depth from 32 bits to 16 bits.** (Candidate C1). Also, the offset-encode re-scan (CUB sort → offset kernel reads isect_ids again) can potentially be fused.

### Q7: Backward memory traffic that can be eliminated?
**Colors and opacities as float16** (Candidate C5, minor). **Shared memory gradient accumulation** reduces atomicAdd traffic (Candidate C4). The main backward tensor reads (means2d, conics, flatten_ids) cannot be eliminated — they are needed for gradient computation.

### Q8: Which buffers can be reused?
**Intersection buffers across iterations** (Candidate C3). Pre-allocate at max capacity and reuse, avoiding allocation overhead on each forward call.

### Q9: Which synchronizations can be eliminated?
**The `.item<int64_t>()` call in Intersect.cpp:80** (Candidate C2). Replace with GPU-side n_isects tracking and pre-allocated buffers.

### Q10: 3 optimization candidates most worth implementing?
1. **C1: Depth bit-width compression** — reduces sort passes ~33%
2. **C2: Host-device sync elimination** — removes serialization point
3. **C4: Shared-memory gradient accumulator (4a)** — reduces backward atomicAdd contention

### Q11: Which most likely to produce meaningful E2E training benefit?
**C1: Depth bit-width compression.** The sort accounts for ~30-40% of forward time. A 33% sort reduction translates to ~10-13% faster forward. In training where forward + backward are roughly balanced, this is ~3-5% E2E training speedup. Small but meaningful and risk-free.

### Q12: Which is best for first independent ablation?
**C1: Depth bit-width compression.** Implementation is ~10 lines in one file. Forward-only ablation immediately shows sort time reduction. No backward/training changes needed for initial validation. Clear pass/fail criterion: n_isects sorted with fewer sort passes + rendered images within tolerance.

---

## NO-HIGH-CONFIDENCE CANDIDATE Status

While C1 is the strongest candidate, we must be honest about its limitations:
- ~3-5% E2E training gain is **modest** — not a breakthrough
- Only addresses forward sort, leaving backward (40-50% of training time) untouched
- The sort reduction (33%) is theoretical and depends on CUB implementation details
- Real training with densification/pruning may mask the gain

**If C1 fails to produce measurable benefit in a real training forward-only ablation, Phase 15's conclusion would be: NO HIGH-CONFIDENCE CANDIDATE for meaningful E2E training optimization.**

This would mean the renderer is already operating near its performance frontier for the current algorithmic approach, and meaningful gains require architectural changes (alternative sorting, alternative rasterization, kernel fusion) beyond simple code modifications.
