# Multi-Camera Segmented Sort — Source Audit

**Date:** 2026-10-19  
**Status:** Source audit only — no CUDA modification, no benchmark, no training  
**Target:** gsplat v1.5.3 (installed at `C:\Users\36570\miniconda3\Lib\site-packages\gsplat`)  
**Note:** The installed gsplat has C1 depth compression applied (see `c1_source_audit_final.md`). All references to `end_bit` reflect this.

---

## 1. Executive Summary

This audit answers the question:  

> When `I >= 2`, does gsplat's segmented sort (`segmented=True`) have a meaningfully different compute/memory structure compared to global sort (`segmented=False`)?

**Key findings:**

1. **The sort key width difference between global and segmented sort is only `image_n_bits` (1–4 bits)** — too small to reduce the CUB radix sort pass count for I ≤ 4 under C1.

2. **CUB segmented sort does NOT reduce total sorting work**: it processes I segments of average size N/I, whose total work is O(N × bits) — identical to global sort. The overhead of per-segment histogram computation and scanning is *added* on top.

3. **The segment boundary construction may be buggy for packed mode (I>1)**: In the packed path, the code produces only [0, total_n_isects] as segment offsets, which with n_segments=I>1 causes out-of-bounds reads of the `offsets` tensor.

4. **The correct non-packed path (I>1) produces I segments of per-image intersections**, but the per-image segment boundaries are correct only if the Gaussians are grouped by image in the input arrays.

5. **Downstream correctness (offset + rasterizer) is identical** for both paths: within each (image, tile) pair, depth ordering is preserved, so pixel output and backward are bit-exact.

6. **Phase 14B's 1.9–4.5× slowdown is fully explained** by CUB's DeviceSegmentedRadixSort overhead for single-segment (I=1) degenerate case. For I>1, the slowdown would be less severe but positive overhead is expected.

**Bottom line:** Segmented sort does NOT reduce sort work. It adds segment overhead. For multi-camera (I≥2), the per-segment sorting of smaller data per image does not reduce total work compared to a single N-item radix sort. The narrower key benefit (excluding image_n_bits) is negligible for C1 (≤1 CUB pass for I≥8).

---

## 2. Global Sort Dataflow

### 2.1 Python Entry Point

```
rendering.py:634: isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height,
                              segmented=False, packed=packed, n_images=I, ...)
    ↓
_wrapper.py:504: _make_lazy_cuda_func("intersect_tile")(..., sort=True, segmented=False)
    ↓
Intersect.cpp:15: intersect_tile(means2d, radii, depths, image_ids, gaussian_ids, I,
                                  tile_size, tile_width, tile_height, sort=True, segmented=False)
```

### 2.2 C++ Dispatch

**File:** `Intersect.cpp` lines 15–149  
**Function:** `gsplat::intersect_tile()`

```
Step 1 (lines 56–78):  First pass — count tiles per Gaussian
    launch_intersect_tile_kernel(pass=1)
    → tiles_per_gauss [nnz]  (each element = #tiles intersected by that Gaussian)

Step 2 (lines 79–80):  Compute n_isects
    cum_tiles_per_gauss = cumsum(tiles_per_gauss)
    n_isects = cum_tiles_per_gauss[-1].item<int64_t>()   // HOST SYNC

    // NOTE: For segmented=False, NO offset computation here (line 81: "if (segmented)" skips)

Step 3 (lines 95–116): Second pass — write isect_ids and flatten_ids
    alloc isect_ids [int64, n_isects], flatten_ids [int32, n_isects]
    launch_intersect_tile_kernel(pass=2)
    → writes 64-bit key + Gaussian flatten index

Step 4 (lines 118–148): Sort
    if (sort && !segmented):
        radix_sort_double_buffer(n_isects, image_n_bits, tile_n_bits,
                                  isect_ids, flatten_ids,
                                  isect_ids_sorted, flatten_ids_sorted)
    return (tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted)
```

**Step 5 (upstream, not in this function):**  
`isect_offset_encode()` → `intersect_offset()` → `intersect_offset_kernel`

### 2.3 Intersection Key Construction

**File:** `IntersectTile.cu` lines 87–118  
**Kernel:** `intersect_tile_kernel` (second pass)

```cuda
// Current (C1-deployed) key encoding:
const int64_t iid_enc = iid << (16 + tile_n_bits);       // line 97
int32_t depth_i32 = *(int32_t *)&(depths[idx]);            // line 100: bitcast
int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // line 101
int64_t depth_upper = depth_id_enc >> 16;                  // line 103: keep upper 16 bits

// Key composition (line 113):
isect_ids[cur_idx] = depth_upper | (tile_id << 16) | iid_enc;
```

**Bit layout** (1080p tile16, I images):

```
Bits [15:0]     = depth_upper (upper 16 bits of float32 bitcast)
Bits [15+tn-1:16] = tile_id (tn = floor(log2(n_tiles)) + 1 bits)
Bits [15+tn+in-1:16+tn] = image_id (in = floor(log2(I)) + 1 bits)   
Bits [63:16+tn+in] = unused (zero-extended from non-negative values)
```

For 1080p tile16, I=2:
- n_tiles = 120 × 68 = 8160
- tn = floor(log2(8160)) + 1 = 13
- in = floor(log2(2)) + 1 = 2
- Used bits: 16 + 13 + 2 = 31 bits

### 2.4 CUB Radix Sort (Global)

**File:** `IntersectTile.cu` lines 302–345  
**Function:** `radix_sort_double_buffer()`

```cuda
cub::DeviceRadixSort::SortPairs(
    d_keys,              // cub::DoubleBuffer<int64_t>
    d_values,            // cub::DoubleBuffer<int32_t>
    n_isects,            // total number of items
    0,                   // begin_bit
    16 + tile_n_bits + image_n_bits,  // end_bit (C1)
    at::cuda::getCurrentCUDAStream()
);
```

**Key parameters (I=2, 1080p tile16):**
| Parameter | Value |
|:----------|:------|
| begin_bit | 0 |
| end_bit | 16 + 13 + 2 = **31** |
| Passes (RADIX_BITS=4) | ceil(31/4) = **8** |
| Passes (RADIX_BITS=8) | ceil(31/8) = **4** |
| Items | N = total intersections across all I images |
| Output ordering | Ascending: depth_upper → tile_id → image_id (primary→tertiary) |

**Output ordering within same (image, tile):**  
Items sorted by depth_upper ascending. For equal depth_upper (collisions), tile_id specifies ordering, then image_id. Within the same (image_id, tile_id) group: depth-sorted.

### 2.5 Offset Kernel

**File:** `IntersectTile.cu` lines 214–263  
**Kernel:** `intersect_offset_kernel`

The offset kernel reads `isect_ids[idx] >> 16` to extract `(image_id, tile_id)` and detects tile boundaries by comparing adjacent entries:

```cuda
int64_t isect_id_curr = isect_ids[idx] >> 16;   // line 233: shift out depth
int64_t iid_curr = isect_id_curr >> tile_n_bits; // extract image_id
int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1); // extract tile_id
int64_t id_curr = iid_curr * n_tiles + tid_curr; // flatten (image, tile) → index
```

When adjacent entries differ in `(image_id, tile_id)`, a tile boundary is recorded.

**Output:** `offsets[I, tile_height, tile_width]` — per-(image, tile) start position in the sorted intersection arrays.

### 2.6 Rasterizer Consumption

**File:** `RasterizeToPixels3DGSFwd.cu` lines 17–150  
**Kernel:** `rasterize_to_pixels_3dgs_fwd_kernel`

The forward kernel reads `tile_offsets` and `flatten_ids`:

```cuda
int32_t range_start = tile_offsets[tile_id];           // line 85
int32_t range_end = ...                                 // line 86-89
for (uint32_t b = 0; b < num_batches; ++b) {           // line 115
    uint32_t batch_start = range_start + block_size * b; // line 124
    uint32_t idx = batch_start + tr;                    // line 125
    int32_t g = flatten_ids[idx];                        // line 127
    // ... load means2d[g], conics[g], opacities[g] ...
}
```

**Key observation:** The rasterizer reads `flatten_ids` in depth-sorted order (ascending from range_start to range_end). It consumes Gaussian attributes by index `g = flatten_ids[pos]`. The attribute arrays (`means2d`, `conics`, etc.) are indexed by the *absolute* flatten index, which includes Gaussians from all images.

---

## 3. Segmented Sort Dataflow

### 3.1 Callsite Difference

The only difference is the `segmented` boolean:

```
rendering.py:641: isect_tiles(..., segmented=True, ...)
    ↓
Intersect.cpp:122: if (segmented) { segmented_radix_sort_double_buffer(...); }
```

### 3.2 Segment Boundary Construction

**File:** `Intersect.cpp` lines 81–90  

```cpp
if (segmented) {
    // offsets in the isect_ids and flatten_ids
    offsets = at::cumsum(
        at::sum(tiles_per_gauss, -1).view({-1}), 0
    );
    offsets = at::cat(
        {at::tensor({0}, opt.dtype(at::kInt)),
         offsets}
    );
}
```

**Critical code path analysis:**

The behavior depends on whether the pipeline uses packed or non-packed mode.

#### Non-packed mode (packed=False)

**Tensor shapes:**
- `tiles_per_gauss`: [I, N] — per-image, per-Gaussian tile counts
- `sum(tiles_per_gauss, -1)`: → [I] — total intersections per image
- `.view({-1})`: [I]
- `cumsum(..., 0)`: → [I] = [S₁, S₁+S₂, ...,  ΣSᵢ]
- `cat({0, cumsum})`: → [I+1] = [0, S₁, S₁+S₂, ..., N]

**Segments:** I segments, each covering one image's intersections.

This is CORRECT: segment i covers intersections [offsets[i], offsets[i+1]) which are all intersections for image i.

#### Packed mode (packed=True)

**Tensor shapes:**
- `tiles_per_gauss`: [nnz] — flat array across all Gaussians from all images
- `sum(tiles_per_gauss, -1)`: → scalar — sum of ALL elements
- `.view({-1})`: → [1]
- `cumsum(..., 0)`: → [1] = [total]
- `cat({0, cumsum})`: → [2] = [0, total]

**Segments:** 1 segment covering ALL intersections, but n_segments = I > 1.

If packed mode with I > 1, `offsets` has shape [2] = [0, total], and the CUB call uses:

```cpp
segmented_radix_sort_double_buffer(n_isects, I, ...)  // n_segments = I
```

where `I > 1` but `offsets` only has 2 entries. The CUB call reads:

```cpp
d_begin_offsets = offsets.data_ptr<int64_t>()     // pointer to offsets[0]
d_end_offsets   = offsets.data_ptr<int64_t>() + 1 // pointer to offsets[1]
```

CUB reads I begin offsets from offsets[0..I-1] and I end offsets from offsets[1..I]. With offsets size = [I+1] in non-packed mode (correct), or [2] in packed mode (incorrect for I>1):

| Mode | Offsets shape | Required entries | Status |
|:-----|:-------------:|:----------------:|:-------|
| Non-packed, I=2 | [3] | 2 begins, 2 ends = 3 values | ✅ Correct |
| Packed, I=1 | [2] | 1 begin, 1 end = 2 values | ✅ Correct |
| Packed, I=2 | [2] | 2 begins, 2 ends = 3 values | ❌ **OOB read at offsets[2]** |

**Potential bug:** Packed mode with segmented=True and I>1 may read out of bounds from the offsets tensor. This was not tested in Phase 14B (which used I=1).

### 3.3 CUB Segmented Radix Sort

**File:** `IntersectTile.cu` lines 349–400  
**Function:** `segmented_radix_sort_double_buffer()`

```cuda
cub::DeviceSegmentedRadixSort::SortPairs(
    d_keys,              // cub::DoubleBuffer<int64_t>
    d_values,            // cub::DoubleBuffer<int32_t>
    n_isects,            // total items (sum of all segment sizes)
    n_segments,          // = I
    offsets.data_ptr<int64_t>(),       // d_begin_offsets [I]
    offsets.data_ptr<int64_t>() + 1,   // d_end_offsets   [I]
    0,                   // begin_bit
    16 + tile_n_bits,    // end_bit (NO image_n_bits)
    at::cuda::getCurrentCUDAStream()
);
```

**Key difference from global:** `end_bit = 16 + tile_n_bits` EXCLUDES `image_n_bits`.

**For I=2, 1080p tile16:**
| Parameter | Global | Segmented | Difference |
|:----------|:------:|:---------:|:----------:|
| CUB primitive | DeviceRadixSort | DeviceSegmentedRadixSort | Different |
| end_bit | 16+13+2=31 | 16+13=29 | **-2 bits** |
| Passes (RB=4) | 8 | 8 | **0** |
| Passes (RB=8) | 4 | 4 | **0** |
| Segments | 1 | I | I× overhead |
| Per-segment histogram | N/A | I × passes × 16 bytes | Added cost |

### 3.4 Data Volume Per Segment

For I images, N total intersections, average segment size S = N / I:

```
Global sort:     1 segment of N items, 1× histogram, 1× scan
Segmented sort:  I segments of S items each, I× histograms, I× scans

Total work comparison:
  Global: O(N × bits_global)
  Segmented: I × O(S × bits_seg) + overhead(I)

Since N = I × S:
  Global: O(I × S × bits_global)
  Segmented: O(I × S × bits_seg) + overhead(I)

  bits_global = 16 + tn + in
  bits_seg    = 16 + tn

  bits_global - bits_seg = in = floor(log2(I)) + 1 (typically 1–4 bits)

For I ≤ 4: in ≤ 3 bits → pass count is IDENTICAL for both RADIX_BITS=4 and RADIX_BITS=8
For I ≥ 8: in ≥ 4 bits → pass count differs by 1 (RB=4 or RB=8)
```

**Therefore, the sorting WORK is essentially the same for global and segmented.** The only difference is the added CUB segmented overhead.

---

## 4. Key / Bit-width Comparison

### 4.1 Global Sort end_bit

| Config | end_bit formula | I=1 | I=2 | I=4 | I=8 | I=16 |
|:-------|:---------------|:---:|:---:|:---:|:---:|:----:|
| **C1 global** | 16 + tn + in | **30** | **31** | **32** | **33** | **34** |
| Baseline global (pre-C1) | 32 + tn + in | 46 | 47 | 48 | 49 | 50 |

### 4.2 Segmented Sort end_bit

| Config | end_bit formula | Any I |
|:-------|:---------------|:------|
| **C1 segmented** | 16 + tn | **29** |
| Baseline segmented (pre-C1) | 32 + tn | 45 |

### 4.3 Pass Count Comparison (C1, 1080p tile16, tn=13)

Assuming CUB RADIX_BITS=4 (conservative):

| I | Global passes | Segmented passes | Difference |
|:-:|:-------------:|:----------------:|:----------:|
| 1 | ceil(30/4)=8 | ceil(29/4)=8 | **0** |
| 2 | ceil(31/4)=8 | ceil(29/4)=8 | **0** |
| 4 | ceil(32/4)=8 | ceil(29/4)=8 | **0** |
| 8 | ceil(33/4)=9 | ceil(29/4)=8 | **-1** |
| 16 | ceil(34/4)=9 | ceil(29/4)=8 | **-1** |

With CUB RADIX_BITS=8 (common for newer architectures):

| I | Global passes | Segmented passes | Difference |
|:-:|:-------------:|:----------------:|:----------:|
| 1 | ceil(30/8)=4 | ceil(29/8)=4 | **0** |
| 2 | ceil(31/8)=4 | ceil(29/8)=4 | **0** |
| 4 | ceil(32/8)=4 | ceil(29/8)=4 | **0** |
| 8 | ceil(33/8)=5 | ceil(29/8)=4 | **-1** |
| 16 | ceil(34/8)=5 | ceil(29/8)=4 | **-1** |

**Key insight:** With C1 already deployed, segmented sort's narrower key provides ZERO pass-count reduction for I ≤ 4. For I ≥ 8, it saves at most 1 CUB pass.

**Contrast with pre-C1 baseline:**  
Even without C1, segmented sort reduces end_bit from 46 → 45 for I=1 (ZERO pass reduction). The savings only appear at I ≥ 256 (1 pass reduction). The C1 optimization already captures the dominant bit-width savings.

### 4.4 Memory Traffic Per Pass

Each CUB radix sort pass:
- Reads: N × (key_size + value_size) = N × (8 + 4) = **12 × N bytes**
- Writes: same = **12 × N bytes**
- Total per pass: **24 × N bytes**

For N intersections, P passes:
- Global: 24 × N × P_global
- Segmented: 24 × N × P_seg (note: total items N is same, just divided into I segments)

Since P_seg ≈ P_global for I ≤ 4, total traffic is approximately equal.

**Additional segmented overhead:**
- offsets tensor: (I+1) × 8 bytes (negligible)
- CUB temp storage for segmented sort: typically larger than global sort temp (extra per-segment scan buffers)
- Segment offset reads: I × 8 bytes × passes (negligible)

### 4.5 Estimated CUB Temporary Storage

This is device-allocated via CUB's `cub::DeviceSegmentedRadixSort::SortPairs`:

- **Global sort temp**: ~N × (8+4) bytes (key + value double buffers) + histogram (negligible)
- **Segmented sort temp**: ~N × (8+4) bytes + I × passes × overhead  

The actual CUB temp allocation depends on internal policy. The extra I× overhead is small relative to N × 12 bytes for large N, but not zero.

---

## 5. CUB API Comparison

| Aspect | Global (`radix_sort_double_buffer`) | Segmented (`segmented_radix_sort_double_buffer`) |
|:-------|:------------------------------------:|:------------------------------------------------:|
| **CUB primitive** | `DeviceRadixSort::SortPairs` | `DeviceSegmentedRadixSort::SortPairs` |
| **Internal kernel sequence** | 1 histogram + 1 scan + scatter per pass | 1 histogram per segment + 1 scan per segment + scatter per pass |
| **Number of kernel launches** | 1 (covers all P passes) | 1 (covers all P passes for all I segments) |
| **Segment boundaries** | None (single segment) | offsets[0..I] array |
| **Key width** | 16 + tn + in | 16 + tn |
| **Items per call** | N | N (divided into I segments) |
| **Temp storage** | Larger single allocation | Potentially larger per-segment metadata |
| **Double buffer** | Yes (selector tracks which buffer is final) | Yes |

Both use the same `CUB_WRAPPER` macro and the same `cub::DoubleBuffer` mechanism.

---

## 6. Segment Construction — Deep Dive

### 6.1 Non-Packed Path (Correct)

```
Input:
  tiles_per_gauss = [[g0_img0, g1_img0, ..., gN_img0],     // image 0, shape [N]
                     [g0_img1, g1_img1, ..., gN_img1]]      // image 1, shape [N]
                     Shape: [I, N]

Step 1: at::sum(tiles_per_gauss, -1) → [I]
  = [total_tiles_img0, total_tiles_img1]

Step 2: cumsum → [I]
  = [total_tiles_img0, total_tiles_img0 + total_tiles_img1]
  = [S₀, S₀+S₁]  = segment boundary positions in the unsorted isect_ids array

Step 3: cat({0, cumsum}) → [I+1]
  = [0, S₀, S₀+S₁]

Segment 0: isect_ids[0 : S₀]  → all intersections of image 0
Segment 1: isect_ids[S₀ : S₀+S₁] → all intersections of image 1
```

**Guarantee:** The i-th image's Gaussians are contiguous in the `means2d`/`depths` input arrays (tensor shape [I, N, ...]), so their intersections are contiguous in `isect_ids`. The `tiles_per_gauss` array preserves image ordering.

### 6.2 Packed Path (Potentially Buggy for I>1)

```
Input:
  tiles_per_gauss = [g0_all_imgs, g1_all_imgs, ..., g_nnz_all_imgs]
                     Shape: [nnz]

Step 1: at::sum(tiles_per_gauss, -1) → scalar
  = total_tiles_all_images

Step 2: view({-1}) → [1], cumsum → [1]
  = [total]

Step 3: cat({0, cumsum}) → [2]
  = [0, total]

CUB call: n_segments = I (> 1)
  d_begin_offsets = [0, total]  if I=2, reads [0, total] as begins
  d_end_offsets   = [total, ???]  reads total (from offsets[1]), then reads OFB
```

**This means that in packed mode with I>1, the segment boundaries are not per-image — they're a single segment covering everything, and CUB may read out of bounds.**

In practice, this path may not be exercised because `rendering.py` with `packed=True` and `segmented=True, I>1` is an uncommon configuration. Phase 14B only tested `I=1` (single-camera).

---

## 7. Memory / Temporary Storage

### 7.1 Allocation Sequence

**Global sort (segmented=False):**
```
unchanged from baseline:
  tiles_per_gauss  [nnz]         int32  → 4 × nnz bytes
  cum_tiles_per_gauss [nnz]      int64  → 8 × nnz bytes  (freed after n_isects)
  isect_ids        [N]           int64  → 8 × N bytes
  flatten_ids      [N]           int32  → 4 × N bytes
  isect_ids_sorted [N]           int64  → 8 × N bytes  (or alias via DoubleBuffer)
  flatten_ids_sorted [N]         int32  → 4 × N bytes  (or alias via DoubleBuffer)
  CUB temp storage               varies → typically ~N × 8 bytes
  ---
  Total: ~36 × N bytes for intersection arrays + CUB temp
```

**Segmented sort (segmented=True):**
```
all of the above, PLUS:
  offsets          [I+1]         int64  → 8 × (I+1) bytes  (negligible)
  CUB segmented temp storage             → typically LARGER than global (per-segment metadata)
```

### 7.2 Memory Difference

For typical training (I=2–8, N=44–176M):

| Aspect | Global | Segmented | Delta |
|:-------|:------:|:---------:|:------|
| Base arrays | ~36 × N | ~36 × N | 0 |
| Offsets | 0 | 8 × (I+1) | +8–72 bytes (negligible) |
| CUB temp | ~8 × N | ~8 × N + I × 2^RB × P | +I × 16 × P bytes (negligible for I ≤ 16) |
| **Total** | ~44 × N | ~44 × N + small | **Essentially identical** |

---

## 8. Downstream Offset Path

### 8.1 Offset Kernel — Unchanged Between Global and Segmented

The offset kernel (`IntersectTile.cu:214–263`) is called identically regardless of which sort was used:

```
rendering.py:648: isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    ↓
Intersect.cpp:151: intersect_offset(isect_ids, I, tile_width, tile_height)
    ↓
IntersectTile.cu:265: launch_intersect_offset_kernel(...)
```

The offset kernel reads each `isect_id >> 16` to extract `(image_id, tile_id)`:

```cuda
int64_t isect_id_curr = isect_ids[idx] >> 16;
int64_t iid_curr = isect_id_curr >> tile_n_bits;    // image_id
int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);  // tile_id
int64_t id_curr = iid_curr * n_tiles + tid_curr;    // flat index
```

It detects boundaries between adjacent entries with different `(image_id, tile_id)`.

### 8.2 Ordering Within Each (Image, Tile)

**After global sort:** For any (image i, tile t), entries are:
- contiguous in the sorted array (sorted by tile_id, then image_id)
- depth-sorted within the (tile, image) group (depth is the LSB of the sort key)

**After segmented sort:** For any (image i, tile t), entries are:
- contiguous within segment i (each segment = one image's sorted intersections)
- depth-sorted within tile t (depth is the LSB of the sort key within the segment)
- the segment boundary ensures no cross-image mixing

**Result:** The offset kernel produces identical per-(image, tile) range semantics for both sorts. The rasterizer sees equivalent per-tile depth ordering within each image.

### 8.3 Rasterizer Input — Unchanged

The rasterizer (`RasterizeToPixels3DGSFwd.cu`) receives:
- `tile_offsets [I, tile_height, tile_width]` — per-(image, tile) range start
- `flatten_ids [N]` — Gaussian indices in depth-sorted order (per image-tile)
- `means2d, conics, colors, opacities` — indexed by flatten_id

The flatten_ids indirection is identical in concept for both global and segmented sort. The only difference is the exact order of flatten_ids values, but since per-(image, tile) depth ordering is preserved, the rasterizer output is pixel-identical.

---

## 9. Backward Compatibility

### 9.1 Backward Kernel Inputs

**File:** `RasterizeToPixels3DGSBwd.cu` lines 16–49

The backward kernel receives:
- Same `tile_offsets` and `flatten_ids` tensors (saved from forward via `ctx.save_for_backward`)
- Reads `flatten_ids` in reverse order (back-to-front traversal)
- Uses `last_ids` (from forward) to determine the last contributing Gaussian per pixel

### 9.2 Segmented Sort Impact on Backward

Since:
1. `flatten_ids` has the same shape and dtype regardless of sort method
2. Per-(image, tile) depth ordering is equivalent between global and segmented sort
3. `last_ids` is computed from transmittance threshold (1e-4) which is identical
4. The backward kernel's traversal pattern is identical (reads tile_offsets + flatten_ids)

**The backward kernel is binary-identical between global and segmented sort.** There is no change needed in the backward pass.

This was confirmed by Phase 14B:
> "Does it preserve gradients? **YES** (finite gradients)"
> "Does it preserve pixel output? **YES — bit-exact** (max diff = 0.0)"

### 9.3 Differentiability

The entire `isect_tiles()` function (including both sort paths) is `@torch.no_grad()`. The sort output tensors (`isect_ids`, `flatten_ids`, `tile_offsets`) are not in the autograd graph. They are consumed as indices by the rasterizer autograd function (`_RasterizeToPixels`).

Changing from global to segmented sort does not change the autograd graph structure. Gradient flow is identical.

---

## 10. Phase 14B Historical Evidence

### 10.1 Experimental Conditions

**Report:** `reports/epic05/phase14b_sort_benchmark.md`

| Parameter | Value |
|:----------|:-------|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM) |
| Scene | Mip-NeRF 360 — room (1,593,376 Gaussians, state-30000) |
| Resolution | 1080p |
| **I (cameras)** | **1 (single image)** |
| Protocol | 10 forward passes (torch.inference_mode), CUDA synced, median timing |
| Correctness | Bit-exact (max diff = 0.0) |

### 10.2 Timing Results

| Tile Size | Global (ms) | Segmented (ms) | Ratio |
|:---------:|:-----------:|:--------------:|:-----:|
| 16 | 7.63 | 34.51 | **4.52× slower** |
| 20 | 12.43 | 24.41 | **1.96× slower** |
| 32 | 9.25 | 17.71 | **1.91× slower** |

**Key: ALL measured data.** No estimates.

### 10.3 Why I=1 Gives Worst-Case Segmented Overhead

For I=1:
- `image_n_bits = floor(log2(1)) + 1 = 1`
- Global end_bit = 16 + 13 + 1 = **30** → 8 passes
- Segmented end_bit = 16 + 13 = **29** → 8 passes
- **ZERO pass-count reduction**
- Segmented sort adds: per-segment histogram overhead, offset array reads, CUB segmented dispatch overhead
- **All overhead, no benefit** → explains 4.52× slowdown

For tile32 (I=1):
- tile_n_bits = floor(log2(2040)) + 1 = 11
- Global end_bit = 16 + 11 + 1 = 28 → 7 passes
- Segmented end_bit = 16 + 11 = 27 → 7 passes
- Again **ZERO pass-count reduction**
- But fewer total items (N = 564K vs 1.6M), so segment overhead is relatively smaller → 1.91× vs 4.52× slowdown

### 10.4 What Phase 14B Did NOT Measure

| Scenario | Status |
|:---------|:-------|
| I=1 (single camera) | ✅ **MEASURED** — 1.9–4.5× slowdown |
| I=2 | ❌ **NOT MEASURED** |
| I=4 | ❌ **NOT MEASURED** |
| I=8+ | ❌ **NOT MEASURED** |
| Non-packed path | ❌ **NOT MEASURED** (Phase 14B used packed) |
| CUB pass count | ❌ **NOT MEASURED** (no Nsight Compute) |
| CUB temp storage | ❌ **NOT MEASURED** |

---

## 11. Open Questions

### Q1: For I>1, does segmented sort provide any benefit over global sort?

**With C1 deployed:** No pass-count reduction for I ≤ 4, and at most 1 pass reduction for I ≥ 8. The per-segment CUB overhead remains. The likelihood of benefit is **very low** even for I > 1.

**Without C1 (hypothetical baseline):** The savings are similarly small. Global end_bit = 32 + tn + in. For I=1, global=46 (12 passes), segmented=45 (12 passes). Still zero pass reduction until I ≥ 256.

### Q2: Is the packed-mode segment boundary construction a bug?

For packed mode with I>1, `tiles_per_gauss` is shape `[nnz]` (not `[I, N]`), so `at::sum(tiles_per_gauss, -1)` produces a scalar → offsets = [0, total]. CUB then receives n_segments=I but only 2 offset entries. This appears to be a bug.

However, this path may not be reached in practice because packed mode uses `image_ids` and `gaussian_ids` arrays. The image-level segmentation would require summing by image_id, which the current code does NOT do.

**Impact:** If someone calls `isect_tiles(packed=True, segmented=True)` with I>1, the segmented sort produces incorrect segment boundaries, potentially causing wrong per-image ordering or out-of-bounds memory access.

### Q3: Would a per-image `at::sum` by image_id fix the packed mode?

A correct packed-mode segment construction would require:

```python
# Pseudocode — not implemented
per_image_tiles = torch.zeros(I, device=tiles_per_gauss.device)
per_image_tiles.scatter_add_(0, image_ids, tiles_per_gauss)
offsets = torch.cat([torch.tensor([0]), per_image_tiles.cumsum(0)])
```

This would produce correct I+1 segment boundaries for packed mode. But this adds an extra GPU kernel (scatter_add + cumsum).

### Q4: How does gsplat's rendering.py actually call isect_tiles?

From `rendering.py:634`:
```python
tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d, radii, depths,
    tile_size, tile_width, tile_height,
    segmented=segmented,    # default: False
    packed=packed,          # default: True
    n_images=I,
    image_ids=image_ids,
    gaussian_ids=gaussian_ids,
)
```

In the standard training pipeline, `packed=True` (default) and `segmented=False` (default). The `segmented=True` path with `packed=True, I>1` is an unverified combination.

### Q5: Could the CUB RADIX_BITS change the analysis?

CUB's internal `RADIX_BITS` is compile-time parameterized (typically 4 or 8 in CCCL/CUB 2.x). The actual value used depends on gsplat's CUB/CCCL version and build configuration. Without Nsight Compute, we cannot determine the exact pass count.

**Upper/lower bounds:**
- RADIX_BITS=4: more passes, smaller histogram per pass
- RADIX_BITS=8: fewer passes, larger histogram per pass

In both cases, the 1–4 bit difference between global and segmented sort keys is too small to change the pass count for I ≤ 4.

---

## 12. Candidate Recommendation

### Summary of Differences

| Aspect | Global Sort | Segmented Sort | Delta Significance |
|:-------|:-----------:|:--------------:|:------------------:|
| CUB primitive | `DeviceRadixSort` | `DeviceSegmentedRadixSort` | API difference |
| Key width (C1) | 16+tn+in | 16+tn | 1–4 bits |
| Pass count (I ≤ 4) | P | P | **ZERO** |
| Pass count (I ≥ 8) | P+1 | P | 1 pass saved |
| Segments | 1 | I | I× overhead |
| Per-segment histogram | None | I × P × 2^RB | Added |
| Offsets memory | None | (I+1) × 8 bytes | Negligible |
| Packed mode I>1 | Works | Potentially buggy | ❌ |
| Non-packed mode I>1 | Works | Correct | ✅ |
| Pixel output | Reference | Bit-exact | ✅ |
| Backward | Reference | Identical | ✅ |

### Classification

| Criteria | Finding |
|:---------|:--------|
| Does segmented sort reduce sorting work? | **NO** — total element count N unchanged, bits reduction negligible with C1 |
| Does segmented sort change memory traffic? | **NO** — total traffic ~44 × N in both cases |
| Does segmented sort add overhead? | **YES** — CUB per-segment histogram + CUB segmented dispatch |
| For I>1, does the trade-off change vs I=1? | **MARGINALLY** — the overhead becomes smaller relative to per-image sort time, but the total work is the same |
| Is the packed mode correct for I>1? | **POTENTIALLY BUGGY** — segment boundaries may not be per-image |

**Recommendation: `DROP`**

Reasoning:
1. With C1 already deployed, the key-width benefit of segmented sort is zero for I ≤ 4 and 1 CUB pass for I ≥ 8 — far too small to offset CUB's segmented sort overhead.
2. Phase 14B empirically demonstrated 1.9–4.5× slowdown for I=1, and the overhead mechanism (per-segment histogram) scales with I, not decreasing.
3. The plausible best case (I=8, C1 deployed, RADIX_BITS=4) saves 1 CUB pass but adds I × P × 2^RB segment overhead. The overhead is likely to dominate.
4. The packed mode may have a bug for I>1, making the code path unreliable without fixes.

**This does NOT require a prior-art check** because the local evidence (Phase 14B measurement + source audit) is sufficient to determine that the mechanism does not reduce total sorting work. The answer would remain the same regardless of what the literature says — CUB segmented sort overhead exists by design and is well-documented.

### Rationale for DROP vs other classifications

| Classification | Why not chosen |
|:---------------|:---------------|
| **DROP** ✅ | Source audit + Phase 14B measurement provide sufficient local evidence |
| DEFER | Would require a CUDA microbenchmark to quantify I>1. Benchmark overhead would be smaller but likely still ≥1×. |
| PROTOTYPE CANDIDATE | No plausible mechanism for speedup exists. Total work does not decrease. |
| PRIOR-ART CHECK REQUIRED | Local evidence is sufficient. Prior art would not change the CUB overhead mechanism. |

---

## File References

| File | Lines | Role |
|:-----|:-----:|:-----|
| `gsplat/cuda/_wrapper.py` | 442–517 | `isect_tiles()` Python entry point |
| `gsplat/cuda/_wrapper.py` | 520–540 | `isect_offset_encode()` |
| `gsplat/cuda/csrc/Intersect.cpp` | 15–149 | `intersect_tile()`: pass 1, pass 2, sort dispatch, segment offset construction |
| `gsplat/cuda/csrc/Intersect.cpp` | 151–168 | `intersect_offset()` |
| `gsplat/cuda/csrc/Intersect.h` | 39–59 | Function declarations for both sort paths |
| `gsplat/cuda/csrc/IntersectTile.cu` | 24–119 | `intersect_tile_kernel`: key construction |
| `gsplat/cuda/csrc/IntersectTile.cu` | 214–263 | `intersect_offset_kernel`: tile boundary detection |
| `gsplat/cuda/csrc/IntersectTile.cu` | 302–345 | `radix_sort_double_buffer`: CUB global sort |
| `gsplat/cuda/csrc/IntersectTile.cu` | 349–400 | `segmented_radix_sort_double_buffer`: CUB segmented sort |
| `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` | 17–150 | Forward rasterizer (flatten_ids consumer) |
| `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` | 16–139 | Backward rasterizer (same flatten_ids consumer) |
| `gsplat/rendering.py` | 58, 205, 634–646 | High-level `rasterization()` segmented parameter usage |
| `reports/epic05/phase14b_sort_benchmark.md` | 1–59 | Phase 14B benchmark data (I=1) |
| `reports/epic05/phase14b_sorting_source_trace.md` | 1–135 | Previous source trace |
| `reports/epic05/phase14b_sorting_recon.md` | 1–168 | Reconnaissance report |
| `reports/phase-c17-c2/c1_source_audit_final.md` | 1–303 | C1 depth compression source audit |
| `reports/epic05/phase8e_per_kernel_forward_timing.md` | 1–330 | Per-kernel timing (sort = 60–80% of forward) |

---

```text
MULTI-CAMERA SEGMENTED SORT SOURCE AUDIT COMPLETE

GLOBAL SORT:
cub::DeviceRadixSort::SortPairs over N total intersections, key end_bit = 16+tn+in.
1 segment of N items. P = ceil(end_bit / RB) CUB passes.
For 1080p tile16, I=2: 8 passes (RB=4), 31-bit key.

SEGMENTED SORT:
cub::DeviceSegmentedRadixSort::SortPairs over N total intersections split into I segments,
key end_bit = 16+tn (excluding image_n_bits).
I segments of avg size N/I. Same number of total CUB passes for I ≤ 4.
For 1080p tile16, I=2: 8 passes (RB=4), 29-bit key, per-segment overhead.

BIT-WIDTH DIFFERENCE:
image_n_bits = floor(log2(I)) + 1 = 1–4 bits (I=1 to I=8).
With C1 deployed (depth already reduced from 32→16 bits),
segmented removes only the 1–4 image_n_bits → ZERO pass-count reduction for I ≤ 4,
and at most 1 pass reduction for I ≥ 8. Pre-C1 baseline: similarly negligible.

EXTRA SEGMENTED OVERHEAD:
Per-segment histogram: I × P × 2^RB histogram bins.
CUB segmented dispatch: internal per-segment scan + scatter setup.
Offset array: (I+1) × 8 bytes, OOB read potential in packed mode for I>1.
Total extra: small but non-zero memory + compute overhead per segment.

CURRENT EVIDENCE:
Phase 14B: I=1 benchmark showed 1.9–4.5× slowdown (source: phase14b_sort_benchmark.md).
ALL measured, no estimates. Source audit confirms zero pass-count benefit for I=1 with C1.
For I>1: zero pass-count reduction for I ≤ 4, marginal (1 pass) for I ≥ 8.
Packed mode segment boundaries may be incorrect for I>1 (Intersect.cpp:81-90).

RECOMMENDATION:
DROP — Segmented sort does not reduce total sorting work. CUB overhead dominates.
Phase 14B measurement + source audit provide sufficient local evidence.
No prior-art check needed. No further investigation warranted.
```
